mod nnls;
mod simd;

use nnls::{BatchWorkspace, solve_nnls_batch};
pub use rng_compat_r::MathMode;
use rng_compat_r::{RRng, pnorm_with_mode};
use simd::{AxpyKernel, select_axpy, select_prediction_error};
use std::fmt;
use std::mem;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidDimensions(&'static str),
    UnsupportedRandomSeed,
}

impl fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(message) => formatter.write_str(message),
            Self::UnsupportedRandomSeed => formatter.write_str("unsupported R random seed"),
        }
    }
}

impl std::error::Error for CoreError {}

pub struct BatchNnlsInput<'a> {
    /// Column-major sample-by-feature matrix.
    pub x: &'a [f64],
    /// Column-major group-by-feature matrix.
    pub means: &'a [f64],
    pub weights: &'a [f64],
    /// Optional column-major sample-by-group warm start.
    pub warm: &'a [f64],
    pub samples: usize,
    pub features: usize,
    pub groups: usize,
}

pub struct GibbsInput<'a> {
    /// Column-major sample-by-feature matrix.
    pub x: &'a [f64],
    /// Column-major group-by-feature matrix of unparameterized medians.
    pub raw_means: &'a [f64],
    pub alpha: &'a [f64],
    pub weight: &'a [f64],
    pub alpha_for_group: &'a [usize],
    pub target_group: &'a [usize],
    pub sample_weight: &'a [f64],
    pub samples: usize,
    pub features: usize,
    pub groups: usize,
    pub iterations: usize,
    pub proposal_sd: f64,
    pub optimize_alpha: bool,
    pub optimize_weight: bool,
    pub math_mode: MathMode,
    pub random_seed: &'a [i32],
}

pub struct GibbsResult {
    pub best_error: f64,
    pub alpha: Vec<f64>,
    pub weight: Vec<f64>,
    /// Row-major group-by-feature basis.
    pub basis: Vec<f64>,
    /// Column-major sample-by-group normalized coefficients.
    pub coefficients: Vec<f64>,
    pub random_seed: Vec<i32>,
}

pub const LECUYER_RANDOM_SEED_LEN: usize = 7;

/// Derive deterministic, non-overlapping R-compatible worker streams.
///
/// Stream zero is `random_seed`. Each subsequent state is advanced by R's
/// `parallel::nextRNGStream()` jump. States are stored contiguously in the
/// returned vector. Importing the initial state preserves the calling R
/// version's serialized RNG metadata.
pub fn lecuyer_stream_seeds(random_seed: &[i32], count: usize) -> Result<Vec<i32>, CoreError> {
    let output_len =
        count
            .checked_mul(LECUYER_RANDOM_SEED_LEN)
            .ok_or(CoreError::InvalidDimensions(
                "random stream count exceeds addressable memory",
            ))?;
    let mut output = vec![0_i32; output_len];
    let mut rng =
        RRng::from_random_seed(random_seed).map_err(|_| CoreError::UnsupportedRandomSeed)?;
    for stream_index in 0..count {
        let offset = stream_index * LECUYER_RANDOM_SEED_LEN;
        rng.write_random_seed(&mut output[offset..offset + LECUYER_RANDOM_SEED_LEN])
            .map_err(|_| CoreError::UnsupportedRandomSeed)?;
        if stream_index + 1 < count {
            rng = rng
                .next_rng_stream()
                .map_err(|_| CoreError::UnsupportedRandomSeed)?;
        }
    }
    Ok(output)
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

fn validate_batch(input: &BatchNnlsInput<'_>) -> Result<(), CoreError> {
    if input.samples == 0 || input.features == 0 || input.groups == 0 || input.groups > 63 {
        return Err(CoreError::InvalidDimensions(
            "NNLS dimensions must be nonzero and groups must not exceed 63",
        ));
    }
    if input.x.len() != input.samples * input.features
        || input.means.len() != input.groups * input.features
        || input.weights.len() != input.features
        || (!input.warm.is_empty() && input.warm.len() != input.samples * input.groups)
    {
        return Err(CoreError::InvalidDimensions(
            "NNLS input slices do not match their declared dimensions",
        ));
    }
    Ok(())
}

pub fn solve_batch(input: BatchNnlsInput<'_>) -> Result<Vec<f64>, CoreError> {
    validate_batch(&input)?;
    let BatchNnlsInput {
        x,
        means,
        weights,
        warm,
        samples,
        features,
        groups,
    } = input;
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
    let mut result = vec![0.0; samples * groups];
    let mut output_masks = vec![0_u64; samples];
    solve_nnls_batch(
        &gram,
        &rhs,
        if warm.is_empty() { &zero_warm } else { warm },
        None,
        samples,
        groups,
        &mut result,
        &mut output_masks,
        &mut BatchWorkspace::new(groups, samples),
    );
    Ok(result)
}

fn validate_gibbs(input: &GibbsInput<'_>) -> Result<(), CoreError> {
    if input.samples == 0 || input.features == 0 || input.groups == 0 || input.groups > 63 {
        return Err(CoreError::InvalidDimensions(
            "Gibbs dimensions must be nonzero and groups must not exceed 63",
        ));
    }
    if input.x.len() != input.samples * input.features
        || input.raw_means.len() != input.groups * input.features
        || input.alpha.len() != input.groups
        || input.weight.len() != input.features
        || input.alpha_for_group.len() != input.groups
        || input.target_group.len() != input.samples
        || input.sample_weight.len() != input.samples
        || input
            .alpha_for_group
            .iter()
            .any(|&index| index >= input.groups)
        || input
            .target_group
            .iter()
            .any(|&index| index >= input.groups)
    {
        return Err(CoreError::InvalidDimensions(
            "Gibbs input slices or indices do not match their declared dimensions",
        ));
    }
    if input.random_seed.is_empty() {
        return Err(CoreError::UnsupportedRandomSeed);
    }
    Ok(())
}

pub fn optimize_gibbs(input: GibbsInput<'_>) -> Result<GibbsResult, CoreError> {
    validate_gibbs(&input)?;
    let GibbsInput {
        x,
        raw_means,
        alpha,
        weight,
        alpha_for_group,
        target_group,
        sample_weight,
        samples,
        features,
        groups,
        iterations,
        proposal_sd,
        optimize_alpha,
        optimize_weight,
        math_mode,
        random_seed,
    } = input;
    let mut rng =
        RRng::from_random_seed(random_seed).map_err(|_| CoreError::UnsupportedRandomSeed)?;
    rng.set_math_mode(math_mode);
    let group_for_alpha: Vec<usize> = (0..groups)
        .map(|alpha_index| {
            alpha_for_group
                .iter()
                .position(|&value| value == alpha_index)
                .unwrap_or(alpha_index)
        })
        .collect();
    let axpy = select_axpy();
    let prediction_error = select_prediction_error();

    let mut current_alpha = alpha.to_vec();
    let mut current_weight = weight.to_vec();
    let mut current_basis = vec![0.0; groups * features];
    let mut current_gram = vec![0.0; groups * groups];
    let mut current_rhs = vec![0.0; samples * groups];
    let mut current_coefficients = vec![0.0; samples * groups];
    build_basis(
        raw_means,
        &current_alpha,
        &current_weight,
        alpha_for_group,
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
    let mut workspace = BatchWorkspace::new(groups, samples);
    let zero_warm = vec![0.0; samples * groups];
    let mut current_masks = vec![0_u64; samples];
    solve_nnls_batch(
        &current_gram,
        &current_rhs,
        &zero_warm,
        None,
        samples,
        groups,
        &mut current_coefficients,
        &mut current_masks,
        &mut workspace,
    );
    let mut current_error = prediction_error(
        &current_coefficients,
        target_group,
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
    let mut proposal_masks = current_masks.clone();

    for _ in 0..iterations {
        proposal_alpha.clone_from_slice(&current_alpha);
        proposal_weight.clone_from_slice(&current_weight);
        proposal_basis.clone_from_slice(&current_basis);
        proposal_rhs.clone_from_slice(&current_rhs);

        let alpha_index = if optimize_alpha {
            let index = rng.sample_index(groups);
            let noise = rng.rnorm(0.0, 1.0);
            proposal_alpha[index] = (proposal_alpha[index]
                + noise * (proposal_alpha[index].abs() + 0.001) * proposal_sd
                + alpha_momentum[index])
                .max(0.0);
            Some(index)
        } else {
            None
        };
        let weight_index = if optimize_weight {
            let index = rng.sample_index(features);
            let noise = rng.rnorm(0.0, 1.0);
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
        solve_nnls_batch(
            &proposal_gram,
            &proposal_rhs,
            &current_coefficients,
            Some(&current_masks),
            samples,
            groups,
            &mut proposal_coefficients,
            &mut proposal_masks,
            &mut workspace,
        );
        let proposal_error = prediction_error(
            &proposal_coefficients,
            target_group,
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
        let probability = 1.0
            - pnorm_with_mode(
                proposal_error,
                current_error,
                current_error / 1000.0,
                true,
                false,
                math_mode,
            );
        if rng.runif() < probability {
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
            mem::swap(&mut current_masks, &mut proposal_masks);
        }
    }

    let mut normalized_coefficients = vec![0.0; samples * groups];
    for sample in 0..samples {
        let mut total = 0.0;
        for group in 0..groups {
            total += best_coefficients[group * samples + sample];
        }
        for group in 0..groups {
            normalized_coefficients[group * samples + sample] =
                best_coefficients[group * samples + sample] / total;
        }
    }
    let mut next_seed = vec![0_i32; rng.random_seed_len()];
    rng.write_random_seed(&mut next_seed)
        .map_err(|_| CoreError::UnsupportedRandomSeed)?;
    Ok(GibbsResult {
        best_error,
        alpha: best_alpha,
        weight: best_weight,
        basis: best_basis,
        coefficients: normalized_coefficients,
        random_seed: next_seed,
    })
}

/// Selected SIMD kernel: scalar=0, NEON=1, AVX2=2, AVX-512=3.
pub fn simd_level() -> i32 {
    simd::kernel_level()
}

#[cfg(test)]
mod tests {
    use super::{LECUYER_RANDOM_SEED_LEN, lecuyer_stream_seeds};
    use rng_compat_r::{RRng, RUniformKind};

    #[test]
    fn worker_streams_match_successive_r_jumps() {
        let initial = RRng::from_seed_with_kind(42, RUniformKind::LecuyerCmrg);
        let streams = lecuyer_stream_seeds(&initial.random_seed(), 4).unwrap();
        assert_eq!(streams.len(), 4 * LECUYER_RANDOM_SEED_LEN);

        let mut expected = initial;
        for stream in streams.chunks_exact(LECUYER_RANDOM_SEED_LEN) {
            assert_eq!(stream, expected.random_seed());
            expected = expected.next_rng_stream().unwrap();
        }
    }
}
