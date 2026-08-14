use std::collections::HashMap;

const ACTIVE_TOLERANCE: f64 = 1.0e-12;

struct PassiveFactor {
    indices: Vec<usize>,
    lu: Vec<f64>,
    pivots: Vec<usize>,
    valid: bool,
    usable: bool,
}

impl PassiveFactor {
    fn for_mask(passive: u64, size: usize) -> Self {
        let indices: Vec<usize> = (0..size)
            .filter(|index| passive & (1_u64 << index) != 0)
            .collect();
        let count = indices.len();
        Self {
            indices,
            lu: vec![0.0; count * count],
            pivots: vec![0; count],
            valid: false,
            usable: false,
        }
    }

    fn new(gram: &[f64], passive: u64, size: usize) -> Option<Self> {
        let mut factor = Self::for_mask(passive, size);
        factor.prepare(gram, size);
        factor.usable.then_some(factor)
    }

    fn prepare(&mut self, gram: &[f64], size: usize) {
        self.valid = true;
        self.usable = false;
        let count = self.indices.len();
        for row in 0..count {
            for column in 0..count {
                self.lu[row * count + column] =
                    gram[self.indices[row] * size + self.indices[column]];
            }
        }
        for pivot in 0..count {
            let mut best = pivot;
            let mut best_value = self.lu[pivot * count + pivot].abs();
            for row in (pivot + 1)..count {
                let candidate = self.lu[row * count + pivot].abs();
                if candidate > best_value {
                    best = row;
                    best_value = candidate;
                }
            }
            if best_value <= f64::EPSILON {
                return;
            }
            self.pivots[pivot] = best;
            if best != pivot {
                // The solve replays pivoting and elimination in the same order,
                // so multipliers from completed columns must remain in place.
                for column in pivot..count {
                    self.lu.swap(pivot * count + column, best * count + column);
                }
            }
            let diagonal = self.lu[pivot * count + pivot];
            for row in (pivot + 1)..count {
                let factor = self.lu[row * count + pivot] / diagonal;
                self.lu[row * count + pivot] = factor;
                for column in (pivot + 1)..count {
                    self.lu[row * count + column] -= factor * self.lu[pivot * count + column];
                }
            }
        }
        self.usable = true;
    }

    fn solve(&self, rhs: &[f64], out: &mut [f64], work: &mut [f64]) {
        out.fill(0.0);
        let count = self.indices.len();
        for row in 0..count {
            work[row] = rhs[self.indices[row]];
        }
        for pivot in 0..count {
            if self.pivots[pivot] != pivot {
                work.swap(pivot, self.pivots[pivot]);
            }
            for row in (pivot + 1)..count {
                work[row] -= self.lu[row * count + pivot] * work[pivot];
            }
        }
        for row in (0..count).rev() {
            let mut value = work[row];
            for column in (row + 1)..count {
                value -= self.lu[row * count + column] * out[self.indices[column]];
            }
            out[self.indices[row]] = value / self.lu[row * count + row];
        }
    }
}

enum FactorStore {
    Dense(Vec<PassiveFactor>),
    Sparse(HashMap<u64, Option<PassiveFactor>>),
}

pub struct FactorCache {
    size: usize,
    store: FactorStore,
    work: Vec<f64>,
}

impl FactorCache {
    pub fn new(size: usize) -> Self {
        let store = if size <= 10 {
            FactorStore::Dense(
                (0..(1_usize << size))
                    .map(|mask| PassiveFactor::for_mask(mask as u64, size))
                    .collect(),
            )
        } else {
            FactorStore::Sparse(HashMap::new())
        };
        Self {
            size,
            store,
            work: vec![0.0; size],
        }
    }

    pub fn clear(&mut self) {
        match &mut self.store {
            FactorStore::Dense(factors) => factors.iter_mut().for_each(|factor| {
                factor.valid = false;
                factor.usable = false;
            }),
            FactorStore::Sparse(factors) => factors.clear(),
        }
    }

    fn solve(&mut self, gram: &[f64], rhs: &[f64], passive: u64, out: &mut [f64]) -> bool {
        match &mut self.store {
            FactorStore::Dense(slots) => {
                let factor = &mut slots[passive as usize];
                if !factor.valid {
                    factor.prepare(gram, self.size);
                }
                if factor.usable {
                    factor.solve(rhs, out, &mut self.work);
                    true
                } else {
                    false
                }
            }
            FactorStore::Sparse(factors) => {
                let factor = factors
                    .entry(passive)
                    .or_insert_with(|| PassiveFactor::new(gram, passive, self.size));
                match factor {
                    Some(factor) => {
                        factor.solve(rhs, out, &mut self.work);
                        true
                    }
                    None => false,
                }
            }
        }
    }
}

pub fn solve_nnls(
    gram: &[f64],
    rhs: &[f64],
    warm: &[f64],
    out: &mut [f64],
    candidate: &mut [f64],
    cache: &mut FactorCache,
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
            if !cache.solve(gram, rhs, passive, candidate) {
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
                    alpha = if denominator > 0.0 {
                        alpha.min(out[index] / denominator)
                    } else {
                        0.0
                    };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_solve(gram: &[f64], rhs: &[f64], passive: u64, size: usize) -> Vec<f64> {
        let indices: Vec<usize> = (0..size)
            .filter(|index| passive & (1_u64 << index) != 0)
            .collect();
        let count = indices.len();
        let mut matrix = vec![0.0; count * (count + 1)];
        for row in 0..count {
            for column in 0..count {
                matrix[row * (count + 1) + column] = gram[indices[row] * size + indices[column]];
            }
            matrix[row * (count + 1) + count] = rhs[indices[row]];
        }
        for pivot in 0..count {
            let mut best = pivot;
            for row in (pivot + 1)..count {
                if matrix[row * (count + 1) + pivot].abs()
                    > matrix[best * (count + 1) + pivot].abs()
                {
                    best = row;
                }
            }
            if best != pivot {
                for column in pivot..=count {
                    matrix.swap(pivot * (count + 1) + column, best * (count + 1) + column);
                }
            }
            for row in (pivot + 1)..count {
                let factor =
                    matrix[row * (count + 1) + pivot] / matrix[pivot * (count + 1) + pivot];
                for column in (pivot + 1)..=count {
                    matrix[row * (count + 1) + column] -=
                        factor * matrix[pivot * (count + 1) + column];
                }
            }
        }
        let mut answer = vec![0.0; size];
        for row in (0..count).rev() {
            let mut value = matrix[row * (count + 1) + count];
            for column in (row + 1)..count {
                value -= matrix[row * (count + 1) + column] * answer[indices[column]];
            }
            answer[indices[row]] = value / matrix[row * (count + 1) + row];
        }
        answer
    }

    #[test]
    fn finds_nonnegative_solution() {
        let gram = [2.0, 1.0, 1.0, 2.0];
        let rhs = [1.0, 1.0];
        let mut out = [0.0; 2];
        solve_nnls(
            &gram,
            &rhs,
            &[0.0; 2],
            &mut out,
            &mut [0.0; 2],
            &mut FactorCache::new(2),
        );
        assert!((out[0] - 1.0 / 3.0).abs() < 1.0e-12);
        assert!((out[1] - 1.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn clamps_bound_variable() {
        let gram = [1.0, 0.0, 0.0, 1.0];
        let rhs = [2.0, -3.0];
        let mut out = [0.0; 2];
        solve_nnls(
            &gram,
            &rhs,
            &[0.0; 2],
            &mut out,
            &mut [0.0; 2],
            &mut FactorCache::new(2),
        );
        assert_eq!(out, [2.0, 0.0]);
    }

    #[test]
    fn cached_factors_match_direct_solves() {
        let size = 7;
        let mut gram = vec![0.0; size * size];
        for row in 0..size {
            for column in 0..size {
                for feature in 0..11 {
                    let left = ((row + 2) * (feature + 3)) as f64 % 13.0 + row as f64;
                    let right = ((column + 2) * (feature + 3)) as f64 % 13.0 + column as f64;
                    gram[row * size + column] += left * right;
                }
            }
            gram[row * size + row] += 0.25;
        }
        let rhs: Vec<f64> = (0..size).map(|index| index as f64 * 1.7 - 2.0).collect();
        let mut cache = FactorCache::new(size);
        for mask in 1..(1_u64 << size) {
            let expected = reference_solve(&gram, &rhs, mask, size);
            let mut actual = vec![0.0; size];
            assert!(cache.solve(&gram, &rhs, mask, &mut actual));
            for index in 0..size {
                assert!(
                    (actual[index] - expected[index]).abs() < 1.0e-9,
                    "mask={mask} index={index} actual={} expected={}",
                    actual[index],
                    expected[index]
                );
            }
        }
    }
}
