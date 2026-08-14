use std::ffi::{c_int, c_void};
use std::slice;

type Sexp = *mut c_void;

const REALSXP: c_int = 14;
const ACTIVE_TOLERANCE: f64 = 1.0e-12;

unsafe extern "C" {
    fn Rf_allocMatrix(kind: c_int, rows: c_int, columns: c_int) -> Sexp;
    fn Rf_ncols(value: Sexp) -> c_int;
    fn Rf_nrows(value: Sexp) -> c_int;
    fn REAL(value: Sexp) -> *mut f64;
    fn Rf_protect(value: Sexp) -> Sexp;
    fn Rf_unprotect(count: c_int);
}

#[inline]
fn dot_column_major(
    left: &[f64],
    left_row: usize,
    right: &[f64],
    right_row: usize,
    rows: usize,
    columns: usize,
) -> f64 {
    let mut total = 0.0;
    for column in 0..columns {
        total += left[left_row + column * rows] * right[right_row + column * rows];
    }
    total
}

fn solve_passive(
    gram: &[f64],
    rhs: &[f64],
    passive: u64,
    size: usize,
    out: &mut [f64],
    indices: &mut [usize],
    matrix: &mut [f64],
) -> bool {
    out.fill(0.0);
    let mut count = 0;
    for index in 0..size {
        if passive & (1_u64 << index) != 0 {
            indices[count] = index;
            count += 1;
        }
    }
    if count == 0 {
        return true;
    }

    // The number of ancestry groups is small. A stack-like dense solve is
    // faster here than crossing another library boundary for every sample.
    let stride = size + 1;
    for row in 0..count {
        let source_row = indices[row];
        for column in 0..count {
            let source_column = indices[column];
            matrix[row * stride + column] = gram[source_row * size + source_column];
        }
        matrix[row * stride + count] = rhs[source_row];
    }

    for pivot in 0..count {
        let mut best = pivot;
        let mut best_value = matrix[pivot * stride + pivot].abs();
        for row in (pivot + 1)..count {
            let candidate = matrix[row * stride + pivot].abs();
            if candidate > best_value {
                best = row;
                best_value = candidate;
            }
        }
        if best_value <= f64::EPSILON {
            return false;
        }
        if best != pivot {
            for column in pivot..=count {
                matrix.swap(pivot * stride + column, best * stride + column);
            }
        }

        let diagonal = matrix[pivot * stride + pivot];
        for row in (pivot + 1)..count {
            let factor = matrix[row * stride + pivot] / diagonal;
            for column in (pivot + 1)..=count {
                matrix[row * stride + column] -= factor * matrix[pivot * stride + column];
            }
        }
    }

    for row in (0..count).rev() {
        let mut value = matrix[row * stride + count];
        for column in (row + 1)..count {
            value -= matrix[row * stride + column] * out[indices[column]];
        }
        out[indices[row]] = value / matrix[row * stride + row];
    }
    true
}

fn nnls(
    gram: &[f64],
    rhs: &[f64],
    warm: &[f64],
    out: &mut [f64],
    candidate: &mut [f64],
    indices: &mut [usize],
    matrix: &mut [f64],
) {
    let size = rhs.len();
    let mut passive = 0_u64;
    for (index, &value) in warm.iter().enumerate() {
        if value > ACTIVE_TOLERANCE {
            passive |= 1_u64 << index;
        }
        out[index] = value.max(0.0);
    }

    let iteration_limit = size * size * 4 + 1;

    for _ in 0..iteration_limit {
        if passive != 0 {
            if !solve_passive(gram, rhs, passive, size, candidate, indices, matrix) {
                // A singular passive set cannot improve the fit reliably.
                let drop_index = (0..size)
                    .filter(|index| passive & (1_u64 << index) != 0)
                    .min_by(|&left, &right| {
                        gram[left * size + left].total_cmp(&gram[right * size + right])
                    })
                    .unwrap_or(0);
                passive &= !(1_u64 << drop_index);
                out[drop_index] = 0.0;
                continue;
            }

            let mut alpha = 1.0_f64;
            let mut needs_boundary_step = false;
            for index in 0..size {
                if passive & (1_u64 << index) != 0 && candidate[index] <= ACTIVE_TOLERANCE {
                    needs_boundary_step = true;
                    let denominator = out[index] - candidate[index];
                    if denominator > 0.0 {
                        alpha = alpha.min(out[index] / denominator);
                    } else {
                        alpha = 0.0;
                    }
                }
            }

            if needs_boundary_step {
                for index in 0..size {
                    out[index] += alpha * (candidate[index] - out[index]);
                    if passive & (1_u64 << index) != 0 && out[index] <= ACTIVE_TOLERANCE {
                        out[index] = 0.0;
                        passive &= !(1_u64 << index);
                    }
                }
                continue;
            }
            out.copy_from_slice(candidate);
        } else {
            out.fill(0.0);
        }

        let mut best_index = None;
        let mut best_gradient = ACTIVE_TOLERANCE;
        for row in 0..size {
            let mut value = rhs[row];
            for column in 0..size {
                value -= gram[row * size + column] * out[column];
            }
            if passive & (1_u64 << row) == 0 && value > best_gradient {
                best_gradient = value;
                best_index = Some(row);
            }
        }

        match best_index {
            Some(index) => passive |= 1_u64 << index,
            None => return,
        }
    }
}

/// Solve all rows of X against the same ancestry basis in one R-to-native call.
///
/// Matrices use R's column-major representation. `means` is already weighted,
/// matching the historical R implementation; `weights` is applied to X here.
///
/// # Safety
///
/// This function must be called by R with double matrices for `x`, `means`, and
/// `warm`, plus a double vector for `weights`. The R wrapper validates matching
/// dimensions before crossing the FFI boundary.
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

    let mut gram = vec![0.0; groups * groups];
    for row in 0..groups {
        for column in 0..groups {
            gram[row * groups + column] =
                dot_column_major(means, row, means, column, groups, features);
        }
    }

    let mut rhs = vec![0.0; groups];
    let mut solution = vec![0.0; groups];
    let mut warm_solution = vec![0.0; groups];
    let mut candidate = vec![0.0; groups];
    let mut indices = vec![0; groups];
    let mut matrix = vec![0.0; groups * (groups + 1)];
    for sample in 0..samples {
        for group in 0..groups {
            let mut total = 0.0;
            for feature in 0..features {
                total += x[sample + feature * samples]
                    * weights[feature]
                    * means[group + feature * groups];
            }
            rhs[group] = total;
            warm_solution[group] = if warm.is_empty() {
                0.0
            } else {
                warm[sample + group * samples]
            };
        }
        nnls(
            &gram,
            &rhs,
            &warm_solution,
            &mut solution,
            &mut candidate,
            &mut indices,
            &mut matrix,
        );
        for group in 0..groups {
            result[sample + group * samples] = solution[group];
        }
    }

    unsafe { Rf_unprotect(1) };
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_nonnegative_solution() {
        let gram = [2.0, 1.0, 1.0, 2.0];
        let rhs = [1.0, 1.0];
        let mut out = [0.0; 2];
        nnls(
            &gram,
            &rhs,
            &[0.0; 2],
            &mut out,
            &mut [0.0; 2],
            &mut [0; 2],
            &mut [0.0; 6],
        );
        assert!((out[0] - 1.0 / 3.0).abs() < 1.0e-12);
        assert!((out[1] - 1.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn clamps_bound_variable() {
        let gram = [1.0, 0.0, 0.0, 1.0];
        let rhs = [2.0, -3.0];
        let mut out = [0.0; 2];
        nnls(
            &gram,
            &rhs,
            &[0.0; 2],
            &mut out,
            &mut [0.0; 2],
            &mut [0; 2],
            &mut [0.0; 6],
        );
        assert_eq!(out, [2.0, 0.0]);
    }
}
