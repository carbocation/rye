use rye_core::{BatchNnlsInput, GibbsInput, MathMode, optimize_gibbs, simd_level, solve_batch};
use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

type Sexp = *mut c_void;

const INTSXP: c_int = 13;
const REALSXP: c_int = 14;
const VECSXP: c_int = 19;

unsafe extern "C" {
    static mut R_NilValue: Sexp;

    fn INTEGER(value: Sexp) -> *mut c_int;
    fn Rf_allocMatrix(kind: c_int, rows: c_int, columns: c_int) -> Sexp;
    fn Rf_allocVector(kind: c_int, length: isize) -> Sexp;
    fn Rf_ncols(value: Sexp) -> c_int;
    fn Rf_nrows(value: Sexp) -> c_int;
    fn Rf_protect(value: Sexp) -> Sexp;
    fn Rf_ScalarInteger(value: c_int) -> Sexp;
    fn Rf_unprotect(count: c_int);
    fn Rf_xlength(value: Sexp) -> isize;
    fn REAL(value: Sexp) -> *mut f64;
    fn SET_VECTOR_ELT(vector: Sexp, index: isize, value: Sexp);
}

/// Return the selected SIMD kernel: scalar=0, NEON=1, AVX2=2, AVX-512=3.
#[unsafe(no_mangle)]
pub extern "C" fn rye_simd_level() -> Sexp {
    unsafe { Rf_ScalarInteger(simd_level()) }
}

/// Return the native optimizer ABI version expected by the R wrapper.
#[unsafe(no_mangle)]
pub extern "C" fn rye_optimizer_abi() -> Sexp {
    unsafe { Rf_ScalarInteger(2) }
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
    let Ok(Ok(coefficients)) = catch_unwind(AssertUnwindSafe(|| {
        solve_batch(BatchNnlsInput {
            x,
            means,
            weights,
            warm,
            samples,
            features,
            groups,
        })
    })) else {
        return unsafe { R_NilValue };
    };
    let answer = unsafe { Rf_protect(Rf_allocMatrix(REALSXP, samples as c_int, groups as c_int)) };
    let result = unsafe { slice::from_raw_parts_mut(REAL(answer), samples * groups) };
    result.copy_from_slice(&coefficients);
    unsafe { Rf_unprotect(1) };
    answer
}

/// Run one complete Gibbs optimization attempt with persistent native state.
///
/// # Safety
///
/// R must pass dimensionally compatible double matrices/vectors and an integer
/// `.Random.seed`. The R wrapper validates dimensions, group indices, RNG type,
/// and the 63-group active-mask limit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rye_gibbs_native_v2(
    x_sexp: Sexp,
    raw_means_sexp: Sexp,
    alpha_sexp: Sexp,
    weight_sexp: Sexp,
    alpha_for_group_sexp: Sexp,
    target_group_sexp: Sexp,
    sample_weight_sexp: Sexp,
    controls_sexp: Sexp,
    seed_sexp: Sexp,
) -> Sexp {
    let samples = unsafe { Rf_nrows(x_sexp) as usize };
    let features = unsafe { Rf_ncols(x_sexp) as usize };
    let groups = unsafe { Rf_nrows(raw_means_sexp) as usize };
    let x = unsafe { slice::from_raw_parts(REAL(x_sexp), samples * features) };
    let raw_means = unsafe { slice::from_raw_parts(REAL(raw_means_sexp), groups * features) };
    let alpha = unsafe { slice::from_raw_parts(REAL(alpha_sexp), groups) };
    let weight = unsafe { slice::from_raw_parts(REAL(weight_sexp), features) };
    let alpha_for_group_input =
        unsafe { slice::from_raw_parts(REAL(alpha_for_group_sexp), groups) };
    let target_group_input = unsafe { slice::from_raw_parts(REAL(target_group_sexp), samples) };
    let sample_weight = unsafe { slice::from_raw_parts(REAL(sample_weight_sexp), samples) };
    let controls = unsafe { slice::from_raw_parts(REAL(controls_sexp), 5) };
    let seed_length = unsafe { Rf_xlength(seed_sexp) };
    if seed_length <= 0 {
        return unsafe { R_NilValue };
    }
    let random_seed = unsafe { slice::from_raw_parts(INTEGER(seed_sexp), seed_length as usize) };
    let alpha_for_group: Vec<usize> = alpha_for_group_input
        .iter()
        .map(|&value| value as usize)
        .collect();
    let target_group: Vec<usize> = target_group_input
        .iter()
        .map(|&value| value as usize)
        .collect();
    let math_mode = if controls[4] != 0.0 {
        MathMode::Deterministic
    } else {
        MathMode::Platform
    };
    let Ok(Ok(optimized)) = catch_unwind(AssertUnwindSafe(|| {
        optimize_gibbs(GibbsInput {
            x,
            raw_means,
            alpha,
            weight,
            alpha_for_group: &alpha_for_group,
            target_group: &target_group,
            sample_weight,
            samples,
            features,
            groups,
            iterations: controls[0] as usize,
            proposal_sd: controls[1],
            optimize_alpha: controls[2] != 0.0,
            optimize_weight: controls[3] != 0.0,
            math_mode,
            random_seed,
        })
    })) else {
        return unsafe { R_NilValue };
    };

    let result_length = 1 + groups + features + groups * features + samples * groups;
    let answer = unsafe { Rf_protect(Rf_allocVector(VECSXP, 2)) };
    let optimizer_result = unsafe { Rf_protect(Rf_allocVector(REALSXP, result_length as isize)) };
    let result = unsafe { slice::from_raw_parts_mut(REAL(optimizer_result), result_length) };
    let mut offset = 0;
    result[offset] = optimized.best_error;
    offset += 1;
    result[offset..offset + groups].copy_from_slice(&optimized.alpha);
    offset += groups;
    result[offset..offset + features].copy_from_slice(&optimized.weight);
    offset += features;
    for feature in 0..features {
        for group in 0..groups {
            result[offset + group + feature * groups] = optimized.basis[group * features + feature];
        }
    }
    offset += groups * features;
    result[offset..].copy_from_slice(&optimized.coefficients);

    let seed_result =
        unsafe { Rf_protect(Rf_allocVector(INTSXP, optimized.random_seed.len() as isize)) };
    let seed_output =
        unsafe { slice::from_raw_parts_mut(INTEGER(seed_result), optimized.random_seed.len()) };
    seed_output.copy_from_slice(&optimized.random_seed);
    unsafe {
        SET_VECTOR_ELT(answer, 0, optimizer_result);
        SET_VECTOR_ELT(answer, 1, seed_result);
        Rf_unprotect(3);
    }
    answer
}
