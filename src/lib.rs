mod nnls;
mod simd;

use nnls::{FactorCache, solve_nnls};
use simd::{AxpyKernel, select_axpy};
use std::ffi::{c_int, c_void};
use std::mem;
use std::slice;

type Sexp = *mut c_void;

const REALSXP: c_int = 14;

unsafe extern "C" {
    fn Rf_allocMatrix(kind: c_int, rows: c_int, columns: c_int) -> Sexp;
    fn Rf_allocVector(kind: c_int, length: isize) -> Sexp;
    fn Rf_ncols(value: Sexp) -> c_int;
    fn Rf_nrows(value: Sexp) -> c_int;
    fn Rf_protect(value: Sexp) -> Sexp;
    fn Rf_ScalarInteger(value: c_int) -> Sexp;
    fn Rf_unprotect(count: c_int);
    fn REAL(value: Sexp) -> *mut f64;

    fn GetRNGstate();
    fn PutRNGstate();
    fn R_unif_index(limit: f64) -> f64;
    fn norm_rand() -> f64;
    fn unif_rand() -> f64;
    fn Rf_pnorm5(value: f64, mean: f64, sd: f64, lower_tail: c_int, log: c_int) -> f64;
}

struct SolverWorkspace {
    rhs: Vec<f64>,
    warm: Vec<f64>,
    solution: Vec<f64>,
    candidate: Vec<f64>,
    factors: FactorCache,
}

impl SolverWorkspace {
    fn new(groups: usize) -> Self {
        Self {
            rhs: vec![0.0; groups],
            warm: vec![0.0; groups],
            solution: vec![0.0; groups],
            candidate: vec![0.0; groups],
            factors: FactorCache::new(groups),
        }
    }
}

#[inline]
fn shrunk_mean(value: f64, alpha: f64) -> f64 {
    let distance = 0.5 - value;
    let direction = if value > 0.5 { -1.0 } else { 1.0 };
    value + distance * distance * direction * alpha
}

fn build_basis(
    raw_means: &[f64],
    alpha: &[f64],
    weight: &[f64],
    alpha_for_group: &[usize],
    groups: usize,
    features: usize,
    basis: &mut [f64],
) {
    for group in 0..groups {
        for feature in 0..features {
            basis[group * features + feature] = shrunk_mean(
                raw_means[group + feature * groups],
                alpha[alpha_for_group[group]],
            ) * weight[feature];
        }
    }
}

fn build_gram(basis: &[f64], groups: usize, features: usize, gram: &mut [f64]) {
    for row in 0..groups {
        for column in 0..=row {
            let mut total = 0.0;
            for feature in 0..features {
                total += basis[row * features + feature] * basis[column * features + feature];
            }
            gram[row * groups + column] = total;
            gram[column * groups + row] = total;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rhs(
    x: &[f64],
    basis: &[f64],
    weight: &[f64],
    samples: usize,
    groups: usize,
    features: usize,
    rhs: &mut [f64],
    axpy: AxpyKernel,
) {
    rhs.fill(0.0);
    for group in 0..groups {
        let destination = &mut rhs[group * samples..(group + 1) * samples];
        for feature in 0..features {
            let source = &x[feature * samples..(feature + 1) * samples];
            axpy(
                destination,
                source,
                weight[feature] * basis[group * features + feature],
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_rhs(
    x: &[f64],
    current_basis: &[f64],
    proposal_basis: &[f64],
    current_weight: &[f64],
    proposal_weight: &[f64],
    alpha_group: Option<usize>,
    weight_feature: Option<usize>,
    samples: usize,
    groups: usize,
    features: usize,
    rhs: &mut [f64],
    axpy: AxpyKernel,
) {
    if let Some(group) = alpha_group {
        let destination = &mut rhs[group * samples..(group + 1) * samples];
        for feature in 0..features {
            let old_scale = current_weight[feature] * current_basis[group * features + feature];
            let new_scale = proposal_weight[feature] * proposal_basis[group * features + feature];
            axpy(
                destination,
                &x[feature * samples..(feature + 1) * samples],
                new_scale - old_scale,
            );
        }
    }
    if let Some(feature) = weight_feature {
        let source = &x[feature * samples..(feature + 1) * samples];
        for group in 0..groups {
            if Some(group) == alpha_group {
                continue;
            }
            let old_scale = current_weight[feature] * current_basis[group * features + feature];
            let new_scale = proposal_weight[feature] * proposal_basis[group * features + feature];
            axpy(
                &mut rhs[group * samples..(group + 1) * samples],
                source,
                new_scale - old_scale,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_all(
    gram: &[f64],
    rhs: &[f64],
    warm: &[f64],
    samples: usize,
    groups: usize,
    output: &mut [f64],
    workspace: &mut SolverWorkspace,
) {
    workspace.factors.clear();
    for sample in 0..samples {
        for group in 0..groups {
            workspace.rhs[group] = rhs[group * samples + sample];
            workspace.warm[group] = warm[group * samples + sample];
        }
        solve_nnls(
            gram,
            &workspace.rhs,
            &workspace.warm,
            &mut workspace.solution,
            &mut workspace.candidate,
            &mut workspace.factors,
        );
        for group in 0..groups {
            output[group * samples + sample] = workspace.solution[group];
        }
    }
}

fn prediction_error(
    coefficients: &[f64],
    target_group: &[usize],
    sample_weight: &[f64],
    samples: usize,
    groups: usize,
) -> f64 {
    let mut error = 0.0;
    for sample in 0..samples {
        let mut total = 0.0;
        for group in 0..groups {
            total += coefficients[group * samples + sample];
        }
        let target = coefficients[target_group[sample] * samples + sample] / total;
        error += sample_weight[sample] * (2.0 * (1.0 - target) / groups as f64);
    }
    error
}

fn copy_normalized(coefficients: &[f64], samples: usize, groups: usize, output: &mut [f64]) {
    for sample in 0..samples {
        let mut total = 0.0;
        for group in 0..groups {
            total += coefficients[group * samples + sample];
        }
        for group in 0..groups {
            output[group * samples + sample] = coefficients[group * samples + sample] / total;
        }
    }
}

/// Return the selected SIMD kernel: scalar=0, NEON=1, AVX2=2, AVX-512=3.
#[unsafe(no_mangle)]
pub extern "C" fn rye_simd_level() -> Sexp {
    unsafe { Rf_ScalarInteger(simd::kernel_level()) }
}

/// Solve all rows of X against the same ancestry basis in one R-to-native call.
///
/// # Safety
///
/// R must pass double matrices for `x`, `means`, and `warm`, plus a double
/// vector for `weights`. The R wrapper validates matching dimensions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rye_nnls_batch(
    x_sexp: Sexp,
    means_sexp: Sexp,
    weights_sexp: Sexp,
    warm_sexp: Sexp,
) -> Sexp {
    let samples = unsafe { Rf_nrows(x_sexp) as usize };
    let features = unsafe { Rf_ncols(x_sexp) as usize };
    let groups = unsafe { Rf_nrows(means_sexp) as usize };
    let warm_rows = unsafe { Rf_nrows(warm_sexp) as usize };
    let warm_columns = unsafe { Rf_ncols(warm_sexp) as usize };
    let x = unsafe { slice::from_raw_parts(REAL(x_sexp), samples * features) };
    let means = unsafe { slice::from_raw_parts(REAL(means_sexp), groups * features) };
    let weights = unsafe { slice::from_raw_parts(REAL(weights_sexp), features) };
    let warm = if warm_rows == samples && warm_columns == groups {
        unsafe { slice::from_raw_parts(REAL(warm_sexp), samples * groups) }
    } else {
        &[]
    };
    let answer = unsafe { Rf_protect(Rf_allocMatrix(REALSXP, samples as c_int, groups as c_int)) };
    let result = unsafe { slice::from_raw_parts_mut(REAL(answer), samples * groups) };
    let mut basis = vec![0.0; groups * features];
    for group in 0..groups {
        for feature in 0..features {
            basis[group * features + feature] = means[group + feature * groups];
        }
    }
    let mut gram = vec![0.0; groups * groups];
    let mut rhs = vec![0.0; samples * groups];
    build_gram(&basis, groups, features, &mut gram);
    build_rhs(
        x,
        &basis,
        weights,
        samples,
        groups,
        features,
        &mut rhs,
        select_axpy(),
    );
    let zero_warm = vec![0.0; samples * groups];
    solve_all(
        &gram,
        &rhs,
        if warm.is_empty() { &zero_warm } else { warm },
        samples,
        groups,
        result,
        &mut SolverWorkspace::new(groups),
    );
    unsafe { Rf_unprotect(1) };
    answer
}

/// Run one complete Gibbs optimization attempt with persistent native state.
///
/// # Safety
///
/// R must pass dimensionally compatible double matrices/vectors. The wrapper
/// validates dimensions, group indices, and the 63-group active-mask limit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rye_gibbs_native(
    x_sexp: Sexp,
    raw_means_sexp: Sexp,
    alpha_sexp: Sexp,
    weight_sexp: Sexp,
    alpha_for_group_sexp: Sexp,
    target_group_sexp: Sexp,
    sample_weight_sexp: Sexp,
    controls_sexp: Sexp,
) -> Sexp {
    let samples = unsafe { Rf_nrows(x_sexp) as usize };
    let features = unsafe { Rf_ncols(x_sexp) as usize };
    let groups = unsafe { Rf_nrows(raw_means_sexp) as usize };
    let x = unsafe { slice::from_raw_parts(REAL(x_sexp), samples * features) };
    let raw_means = unsafe { slice::from_raw_parts(REAL(raw_means_sexp), groups * features) };
    let alpha_input = unsafe { slice::from_raw_parts(REAL(alpha_sexp), groups) };
    let weight_input = unsafe { slice::from_raw_parts(REAL(weight_sexp), features) };
    let alpha_for_group_input =
        unsafe { slice::from_raw_parts(REAL(alpha_for_group_sexp), groups) };
    let target_group_input = unsafe { slice::from_raw_parts(REAL(target_group_sexp), samples) };
    let sample_weight = unsafe { slice::from_raw_parts(REAL(sample_weight_sexp), samples) };
    let controls = unsafe { slice::from_raw_parts(REAL(controls_sexp), 4) };
    let iterations = controls[0] as usize;
    let proposal_sd = controls[1];
    let optimize_alpha = controls[2] != 0.0;
    let optimize_weight = controls[3] != 0.0;
    let alpha_for_group: Vec<usize> = alpha_for_group_input
        .iter()
        .map(|&value| value as usize)
        .collect();
    let group_for_alpha: Vec<usize> = (0..groups)
        .map(|alpha_index| {
            alpha_for_group
                .iter()
                .position(|&value| value == alpha_index)
                .unwrap_or(alpha_index)
        })
        .collect();
    let target_group: Vec<usize> = target_group_input
        .iter()
        .map(|&value| value as usize)
        .collect();
    let axpy = select_axpy();

    let mut current_alpha = alpha_input.to_vec();
    let mut current_weight = weight_input.to_vec();
    let mut current_basis = vec![0.0; groups * features];
    let mut current_gram = vec![0.0; groups * groups];
    let mut current_rhs = vec![0.0; samples * groups];
    let mut current_coefficients = vec![0.0; samples * groups];
    build_basis(
        raw_means,
        &current_alpha,
        &current_weight,
        &alpha_for_group,
        groups,
        features,
        &mut current_basis,
    );
    build_gram(&current_basis, groups, features, &mut current_gram);
    build_rhs(
        x,
        &current_basis,
        &current_weight,
        samples,
        groups,
        features,
        &mut current_rhs,
        axpy,
    );
    let mut workspace = SolverWorkspace::new(groups);
    let zero_warm = vec![0.0; samples * groups];
    solve_all(
        &current_gram,
        &current_rhs,
        &zero_warm,
        samples,
        groups,
        &mut current_coefficients,
        &mut workspace,
    );
    let mut current_error = prediction_error(
        &current_coefficients,
        &target_group,
        sample_weight,
        samples,
        groups,
    );

    let mut best_error = current_error;
    let mut best_alpha = current_alpha.clone();
    let mut best_weight = current_weight.clone();
    let mut best_basis = current_basis.clone();
    let mut best_coefficients = current_coefficients.clone();
    let mut alpha_momentum = vec![0.0; groups];
    let mut weight_momentum = vec![0.0; features];

    let mut proposal_alpha = current_alpha.clone();
    let mut proposal_weight = current_weight.clone();
    let mut proposal_basis = current_basis.clone();
    let mut proposal_gram = current_gram.clone();
    let mut proposal_rhs = current_rhs.clone();
    let mut proposal_coefficients = current_coefficients.clone();

    unsafe { GetRNGstate() };
    for _ in 0..iterations {
        proposal_alpha.clone_from_slice(&current_alpha);
        proposal_weight.clone_from_slice(&current_weight);
        proposal_basis.clone_from_slice(&current_basis);
        proposal_rhs.clone_from_slice(&current_rhs);

        let alpha_index = if optimize_alpha {
            let index = unsafe { R_unif_index(groups as f64) as usize };
            let noise = unsafe { norm_rand() };
            proposal_alpha[index] = (proposal_alpha[index]
                + noise * (proposal_alpha[index].abs() + 0.001) * proposal_sd
                + alpha_momentum[index])
                .max(0.0);
            Some(index)
        } else {
            None
        };
        let weight_index = if optimize_weight {
            let index = unsafe { R_unif_index(features as f64) as usize };
            let noise = unsafe { norm_rand() };
            proposal_weight[index] = (proposal_weight[index]
                + noise * (proposal_weight[index] + 0.001) * proposal_sd
                + weight_momentum[index])
                .max(0.0);
            Some(index)
        } else {
            None
        };
        let alpha_group = alpha_index.map(|index| group_for_alpha[index]);

        if let Some(group) = alpha_group {
            for feature in 0..features {
                proposal_basis[group * features + feature] = shrunk_mean(
                    raw_means[group + feature * groups],
                    proposal_alpha[alpha_for_group[group]],
                ) * proposal_weight[feature];
            }
        }
        if let Some(feature) = weight_index {
            for group in 0..groups {
                proposal_basis[group * features + feature] = shrunk_mean(
                    raw_means[group + feature * groups],
                    proposal_alpha[alpha_for_group[group]],
                ) * proposal_weight[feature];
            }
        }
        update_rhs(
            x,
            &current_basis,
            &proposal_basis,
            &current_weight,
            &proposal_weight,
            alpha_group,
            weight_index,
            samples,
            groups,
            features,
            &mut proposal_rhs,
            axpy,
        );
        build_gram(&proposal_basis, groups, features, &mut proposal_gram);
        solve_all(
            &proposal_gram,
            &proposal_rhs,
            &current_coefficients,
            samples,
            groups,
            &mut proposal_coefficients,
            &mut workspace,
        );
        let proposal_error = prediction_error(
            &proposal_coefficients,
            &target_group,
            sample_weight,
            samples,
            groups,
        );
        if proposal_error < best_error {
            best_error = proposal_error;
            best_alpha.clone_from_slice(&proposal_alpha);
            best_weight.clone_from_slice(&proposal_weight);
            best_basis.clone_from_slice(&proposal_basis);
            best_coefficients.clone_from_slice(&proposal_coefficients);
        }
        let probability =
            unsafe { 1.0 - Rf_pnorm5(proposal_error, current_error, current_error / 1000.0, 1, 0) };
        if unsafe { unif_rand() } < probability {
            for index in 0..groups {
                alpha_momentum[index] = alpha_momentum[index] / 2.0
                    + (proposal_alpha[index] - current_alpha[index]) / 10.0;
            }
            for index in 0..features {
                weight_momentum[index] = weight_momentum[index] / 2.0
                    + (proposal_weight[index] - current_weight[index]) / 10.0;
            }
            current_error = proposal_error;
            mem::swap(&mut current_alpha, &mut proposal_alpha);
            mem::swap(&mut current_weight, &mut proposal_weight);
            mem::swap(&mut current_basis, &mut proposal_basis);
            mem::swap(&mut current_gram, &mut proposal_gram);
            mem::swap(&mut current_rhs, &mut proposal_rhs);
            mem::swap(&mut current_coefficients, &mut proposal_coefficients);
        }
    }
    unsafe { PutRNGstate() };

    let result_length = 1 + groups + features + groups * features + samples * groups;
    let answer = unsafe { Rf_protect(Rf_allocVector(REALSXP, result_length as isize)) };
    let result = unsafe { slice::from_raw_parts_mut(REAL(answer), result_length) };
    let mut offset = 0;
    result[offset] = best_error;
    offset += 1;
    result[offset..offset + groups].copy_from_slice(&best_alpha);
    offset += groups;
    result[offset..offset + features].copy_from_slice(&best_weight);
    offset += features;
    for feature in 0..features {
        for group in 0..groups {
            result[offset + group + feature * groups] = best_basis[group * features + feature];
        }
    }
    offset += groups * features;
    copy_normalized(&best_coefficients, samples, groups, &mut result[offset..]);
    unsafe { Rf_unprotect(1) };
    answer
}
