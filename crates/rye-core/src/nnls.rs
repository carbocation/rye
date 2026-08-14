use std::collections::HashMap;

const ACTIVE_TOLERANCE: f64 = 1.0e-12;
const MAX_BATCH_GROUPS: usize = 10;

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
        self.usable = match count {
            1 => self.factorize_fixed::<1>(),
            2 => self.factorize_fixed::<2>(),
            3 => self.factorize_fixed::<3>(),
            _ => self.factorize_generic(),
        };
    }

    #[allow(clippy::needless_range_loop)]
    fn factorize_fixed<const COUNT: usize>(&mut self) -> bool {
        for pivot in 0..COUNT {
            let mut best = pivot;
            let mut best_value = self.lu[pivot * COUNT + pivot].abs();
            for row in (pivot + 1)..COUNT {
                let candidate = self.lu[row * COUNT + pivot].abs();
                if candidate > best_value {
                    best = row;
                    best_value = candidate;
                }
            }
            if best_value <= f64::EPSILON {
                return false;
            }
            self.pivots[pivot] = best;
            if best != pivot {
                // The solve replays pivoting and elimination in the same order,
                // so multipliers from completed columns must remain in place.
                for column in pivot..COUNT {
                    self.lu.swap(pivot * COUNT + column, best * COUNT + column);
                }
            }
            let diagonal = self.lu[pivot * COUNT + pivot];
            for row in (pivot + 1)..COUNT {
                let factor = self.lu[row * COUNT + pivot] / diagonal;
                self.lu[row * COUNT + pivot] = factor;
                for column in (pivot + 1)..COUNT {
                    self.lu[row * COUNT + column] -= factor * self.lu[pivot * COUNT + column];
                }
            }
        }
        true
    }

    fn factorize_generic(&mut self) -> bool {
        let count = self.indices.len();
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
                return false;
            }
            self.pivots[pivot] = best;
            if best != pivot {
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
        true
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

    fn factor(&mut self, gram: &[f64], passive: u64) -> Option<&PassiveFactor> {
        match &mut self.store {
            FactorStore::Dense(slots) => {
                let factor = &mut slots[passive as usize];
                if !factor.valid {
                    factor.prepare(gram, self.size);
                }
                factor.usable.then_some(&*factor)
            }
            FactorStore::Sparse(factors) => factors
                .entry(passive)
                .or_insert_with(|| PassiveFactor::new(gram, passive, self.size))
                .as_ref(),
        }
    }
}

#[derive(Clone, Copy)]
enum BatchKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl BatchKernel {
    fn selected() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Neon;
        }
        #[allow(unreachable_code)]
        Self::Scalar
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_factor(
        self,
        factor: &PassiveFactor,
        gram: &[f64],
        rhs: &[f64],
        samples: usize,
        groups: usize,
        sample_indices: &[usize],
        output: &mut [f64],
        fallback: &mut Vec<usize>,
    ) {
        match self {
            Self::Scalar => solve_factor_scalar(
                factor,
                gram,
                rhs,
                samples,
                groups,
                sample_indices,
                output,
                fallback,
            ),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe {
                x86::solve_factor_avx2(
                    factor,
                    gram,
                    rhs,
                    samples,
                    groups,
                    sample_indices,
                    output,
                    fallback,
                )
            },
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => unsafe {
                x86::solve_factor_avx512(
                    factor,
                    gram,
                    rhs,
                    samples,
                    groups,
                    sample_indices,
                    output,
                    fallback,
                )
            },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe {
                arm::solve_factor_neon(
                    factor,
                    gram,
                    rhs,
                    samples,
                    groups,
                    sample_indices,
                    output,
                    fallback,
                )
            },
        }
    }
}

pub struct BatchWorkspace {
    rhs: Vec<f64>,
    warm: Vec<f64>,
    solution: Vec<f64>,
    candidate: Vec<f64>,
    factors: FactorCache,
    buckets: Vec<Vec<usize>>,
    bucketed_masks: Vec<u64>,
    fallback: Vec<usize>,
    kernel: BatchKernel,
}

impl BatchWorkspace {
    pub fn new(groups: usize, samples: usize) -> Self {
        let bucket_count = if groups <= MAX_BATCH_GROUPS {
            1_usize << groups
        } else {
            0
        };
        Self {
            rhs: vec![0.0; groups],
            warm: vec![0.0; groups],
            solution: vec![0.0; groups],
            candidate: vec![0.0; groups],
            factors: FactorCache::new(groups),
            buckets: (0..bucket_count).map(|_| Vec::new()).collect(),
            bucketed_masks: vec![u64::MAX; samples],
            fallback: Vec::with_capacity(samples / 32 + 16),
            kernel: BatchKernel::selected(),
        }
    }

    fn prepare_buckets(
        &mut self,
        warm: &[f64],
        warm_masks: Option<&[u64]>,
        samples: usize,
        groups: usize,
    ) {
        let mut changed = false;
        for sample in 0..samples {
            let mask = if let Some(masks) = warm_masks.filter(|masks| masks.len() == samples) {
                masks[sample]
            } else {
                let mut mask = 0_u64;
                for group in 0..groups {
                    if warm[group * samples + sample] > ACTIVE_TOLERANCE {
                        mask |= 1_u64 << group;
                    }
                }
                mask
            };
            if self.bucketed_masks[sample] != mask {
                self.bucketed_masks[sample] = mask;
                changed = true;
            }
        }
        if changed {
            self.buckets.iter_mut().for_each(Vec::clear);
            for (sample, &mask) in self.bucketed_masks.iter().enumerate() {
                self.buckets[mask as usize].push(sample);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn solve_nnls_batch(
    gram: &[f64],
    rhs: &[f64],
    warm: &[f64],
    warm_masks: Option<&[u64]>,
    samples: usize,
    groups: usize,
    output: &mut [f64],
    output_masks: &mut [u64],
    workspace: &mut BatchWorkspace,
) {
    workspace.factors.clear();
    output.fill(0.0);

    if groups > MAX_BATCH_GROUPS {
        solve_all_scalar(
            gram,
            rhs,
            warm,
            samples,
            groups,
            output,
            output_masks,
            workspace,
        );
        return;
    }

    workspace.prepare_buckets(warm, warm_masks, samples, groups);
    workspace.fallback.clear();

    let BatchWorkspace {
        factors,
        buckets,
        fallback,
        kernel,
        ..
    } = workspace;
    for (mask, sample_indices) in buckets.iter().enumerate() {
        if sample_indices.is_empty() {
            continue;
        }
        let passive = mask as u64;
        for &sample in sample_indices {
            output_masks[sample] = passive;
        }
        if passive == 0 {
            for &sample in sample_indices {
                let valid =
                    (0..groups).all(|group| rhs[group * samples + sample] <= ACTIVE_TOLERANCE);
                if !valid {
                    fallback.push(sample);
                }
            }
            continue;
        }
        let Some(factor) = factors.factor(gram, passive) else {
            fallback.extend_from_slice(sample_indices);
            continue;
        };
        kernel.solve_factor(
            factor,
            gram,
            rhs,
            samples,
            groups,
            sample_indices,
            output,
            fallback,
        );
    }

    let BatchWorkspace {
        rhs: sample_rhs,
        warm: sample_warm,
        solution,
        candidate,
        factors,
        fallback,
        ..
    } = workspace;
    for &sample in fallback.iter() {
        for group in 0..groups {
            sample_rhs[group] = rhs[group * samples + sample];
            sample_warm[group] = warm[group * samples + sample];
        }
        output_masks[sample] =
            solve_nnls(gram, sample_rhs, sample_warm, solution, candidate, factors);
        for group in 0..groups {
            output[group * samples + sample] = solution[group];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_all_scalar(
    gram: &[f64],
    rhs: &[f64],
    warm: &[f64],
    samples: usize,
    groups: usize,
    output: &mut [f64],
    output_masks: &mut [u64],
    workspace: &mut BatchWorkspace,
) {
    for sample in 0..samples {
        for group in 0..groups {
            workspace.rhs[group] = rhs[group * samples + sample];
            workspace.warm[group] = warm[group * samples + sample];
        }
        output_masks[sample] = solve_nnls(
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

#[allow(clippy::too_many_arguments)]
fn solve_factor_scalar(
    factor: &PassiveFactor,
    gram: &[f64],
    rhs: &[f64],
    samples: usize,
    groups: usize,
    sample_indices: &[usize],
    output: &mut [f64],
    fallback: &mut Vec<usize>,
) {
    for &sample in sample_indices {
        if !solve_factor_sample(factor, gram, rhs, samples, groups, sample, output) {
            fallback.push(sample);
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn solve_factor_sample(
    factor: &PassiveFactor,
    gram: &[f64],
    rhs: &[f64],
    samples: usize,
    groups: usize,
    sample: usize,
    output: &mut [f64],
) -> bool {
    let count = factor.indices.len();
    let mut work = [0.0; MAX_BATCH_GROUPS];
    let mut solution = [0.0; MAX_BATCH_GROUPS];
    for row in 0..count {
        work[row] = rhs[factor.indices[row] * samples + sample];
    }
    for pivot in 0..count {
        if factor.pivots[pivot] != pivot {
            work.swap(pivot, factor.pivots[pivot]);
        }
        for row in (pivot + 1)..count {
            work[row] -= factor.lu[row * count + pivot] * work[pivot];
        }
    }
    for row in (0..count).rev() {
        let mut value = work[row];
        for column in (row + 1)..count {
            value -= factor.lu[row * count + column] * solution[column];
        }
        solution[row] = value / factor.lu[row * count + row];
        if solution[row] <= ACTIVE_TOLERANCE {
            return false;
        }
    }
    for row in 0..groups {
        if factor.indices.contains(&row) {
            continue;
        }
        let mut gradient = rhs[row * samples + sample];
        for column in 0..count {
            gradient -= gram[row * groups + factor.indices[column]] * solution[column];
        }
        if gradient > ACTIVE_TOLERANCE {
            return false;
        }
    }
    for column in 0..count {
        output[factor.indices[column] * samples + sample] = solution[column];
    }
    true
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use std::arch::x86_64::*;

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn solve_factor_avx2(
        factor: &PassiveFactor,
        gram: &[f64],
        rhs: &[f64],
        samples: usize,
        groups: usize,
        sample_indices: &[usize],
        output: &mut [f64],
        fallback: &mut Vec<usize>,
    ) {
        let count = factor.indices.len();
        let chunks = sample_indices.len().div_ceil(4);
        let tolerance = _mm256_set1_pd(ACTIVE_TOLERANCE);
        for chunk in 0..chunks {
            let first = chunk * 4;
            let lane_count = (sample_indices.len() - first).min(4);
            let mut lanes = [sample_indices[first]; 4];
            lanes[..lane_count].copy_from_slice(&sample_indices[first..first + lane_count]);
            let offsets = _mm_set_epi32(
                lanes[3] as i32,
                lanes[2] as i32,
                lanes[1] as i32,
                lanes[0] as i32,
            );
            let mut work = [_mm256_setzero_pd(); MAX_BATCH_GROUPS];
            let mut solution = [_mm256_setzero_pd(); MAX_BATCH_GROUPS];
            for row in 0..count {
                let base = unsafe { rhs.as_ptr().add(factor.indices[row] * samples) };
                work[row] = unsafe { _mm256_i32gather_pd(base, offsets, 8) };
            }
            solve_lu_avx2(factor, &mut work, &mut solution);

            let mut valid = 0b1111_i32;
            for value in solution.iter().take(count) {
                valid &= _mm256_movemask_pd(_mm256_cmp_pd(*value, tolerance, _CMP_GT_OQ));
            }
            for row in 0..groups {
                if factor.indices.contains(&row) {
                    continue;
                }
                let base = unsafe { rhs.as_ptr().add(row * samples) };
                let mut gradient = unsafe { _mm256_i32gather_pd(base, offsets, 8) };
                for (column, value) in solution.iter().take(count).enumerate() {
                    let coefficient = _mm256_set1_pd(gram[row * groups + factor.indices[column]]);
                    gradient = _mm256_sub_pd(gradient, _mm256_mul_pd(coefficient, *value));
                }
                valid &= _mm256_movemask_pd(_mm256_cmp_pd(gradient, tolerance, _CMP_LE_OQ));
            }

            let mut values = [0.0; 4];
            for (column, value) in solution.iter().take(count).enumerate() {
                unsafe { _mm256_storeu_pd(values.as_mut_ptr(), *value) };
                let base = factor.indices[column] * samples;
                for lane in 0..lane_count {
                    output[base + lanes[lane]] = values[lane];
                }
            }
            for (lane, &sample) in lanes.iter().take(lane_count).enumerate() {
                if valid & (1 << lane) == 0 {
                    fallback.push(sample);
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    fn solve_lu_avx2(
        factor: &PassiveFactor,
        work: &mut [__m256d; MAX_BATCH_GROUPS],
        solution: &mut [__m256d; MAX_BATCH_GROUPS],
    ) {
        match factor.indices.len() {
            1 => solve_lu_avx2_fixed::<1>(factor, work, solution),
            2 => solve_lu_avx2_fixed::<2>(factor, work, solution),
            3 => solve_lu_avx2_fixed::<3>(factor, work, solution),
            _ => solve_lu_avx2_generic(factor, work, solution),
        }
    }

    #[target_feature(enable = "avx2")]
    fn solve_lu_avx2_fixed<const COUNT: usize>(
        factor: &PassiveFactor,
        work: &mut [__m256d; MAX_BATCH_GROUPS],
        solution: &mut [__m256d; MAX_BATCH_GROUPS],
    ) {
        for pivot in 0..COUNT {
            if factor.pivots[pivot] != pivot {
                work.swap(pivot, factor.pivots[pivot]);
            }
            let pivot_value = work[pivot];
            for row in (pivot + 1)..COUNT {
                let multiplier = _mm256_set1_pd(factor.lu[row * COUNT + pivot]);
                work[row] = _mm256_sub_pd(work[row], _mm256_mul_pd(multiplier, pivot_value));
            }
        }
        for row in (0..COUNT).rev() {
            let mut value = work[row];
            for column in (row + 1)..COUNT {
                let multiplier = _mm256_set1_pd(factor.lu[row * COUNT + column]);
                value = _mm256_sub_pd(value, _mm256_mul_pd(multiplier, solution[column]));
            }
            solution[row] = _mm256_div_pd(value, _mm256_set1_pd(factor.lu[row * COUNT + row]));
        }
    }

    #[target_feature(enable = "avx2")]
    fn solve_lu_avx2_generic(
        factor: &PassiveFactor,
        work: &mut [__m256d; MAX_BATCH_GROUPS],
        solution: &mut [__m256d; MAX_BATCH_GROUPS],
    ) {
        let count = factor.indices.len();
        for pivot in 0..count {
            if factor.pivots[pivot] != pivot {
                work.swap(pivot, factor.pivots[pivot]);
            }
            let pivot_value = work[pivot];
            for row in (pivot + 1)..count {
                let multiplier = _mm256_set1_pd(factor.lu[row * count + pivot]);
                work[row] = _mm256_sub_pd(work[row], _mm256_mul_pd(multiplier, pivot_value));
            }
        }
        for row in (0..count).rev() {
            let mut value = work[row];
            for column in (row + 1)..count {
                let multiplier = _mm256_set1_pd(factor.lu[row * count + column]);
                value = _mm256_sub_pd(value, _mm256_mul_pd(multiplier, solution[column]));
            }
            solution[row] = _mm256_div_pd(value, _mm256_set1_pd(factor.lu[row * count + row]));
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn solve_factor_avx512(
        factor: &PassiveFactor,
        gram: &[f64],
        rhs: &[f64],
        samples: usize,
        groups: usize,
        sample_indices: &[usize],
        output: &mut [f64],
        fallback: &mut Vec<usize>,
    ) {
        let count = factor.indices.len();
        let chunks = sample_indices.len().div_ceil(8);
        let tolerance = _mm512_set1_pd(ACTIVE_TOLERANCE);
        for chunk in 0..chunks {
            let first = chunk * 8;
            let lane_count = (sample_indices.len() - first).min(8);
            let mut lanes = [sample_indices[first]; 8];
            lanes[..lane_count].copy_from_slice(&sample_indices[first..first + lane_count]);
            let offsets = _mm256_set_epi32(
                lanes[7] as i32,
                lanes[6] as i32,
                lanes[5] as i32,
                lanes[4] as i32,
                lanes[3] as i32,
                lanes[2] as i32,
                lanes[1] as i32,
                lanes[0] as i32,
            );
            let mut work = [_mm512_setzero_pd(); MAX_BATCH_GROUPS];
            let mut solution = [_mm512_setzero_pd(); MAX_BATCH_GROUPS];
            for row in 0..count {
                let base = unsafe { rhs.as_ptr().add(factor.indices[row] * samples) };
                work[row] = unsafe { _mm512_i32gather_pd(offsets, base, 8) };
            }
            solve_lu_avx512(factor, &mut work, &mut solution);

            let mut valid = 0xff_u8;
            for value in solution.iter().take(count) {
                valid &= _mm512_cmp_pd_mask(*value, tolerance, _CMP_GT_OQ);
            }
            for row in 0..groups {
                if factor.indices.contains(&row) {
                    continue;
                }
                let base = unsafe { rhs.as_ptr().add(row * samples) };
                let mut gradient = unsafe { _mm512_i32gather_pd(offsets, base, 8) };
                for (column, value) in solution.iter().take(count).enumerate() {
                    let coefficient = _mm512_set1_pd(gram[row * groups + factor.indices[column]]);
                    gradient = _mm512_sub_pd(gradient, _mm512_mul_pd(coefficient, *value));
                }
                valid &= _mm512_cmp_pd_mask(gradient, tolerance, _CMP_LE_OQ);
            }

            let mut values = [0.0; 8];
            for (column, value) in solution.iter().take(count).enumerate() {
                let base = factor.indices[column] * samples;
                if lane_count == 8 {
                    let destination = unsafe { output.as_mut_ptr().add(base) };
                    unsafe { _mm512_i32scatter_pd(destination, offsets, *value, 8) };
                } else {
                    unsafe { _mm512_storeu_pd(values.as_mut_ptr(), *value) };
                    for lane in 0..lane_count {
                        output[base + lanes[lane]] = values[lane];
                    }
                }
            }
            for (lane, &sample) in lanes.iter().take(lane_count).enumerate() {
                if valid & (1 << lane) == 0 {
                    fallback.push(sample);
                }
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    fn solve_lu_avx512(
        factor: &PassiveFactor,
        work: &mut [__m512d; MAX_BATCH_GROUPS],
        solution: &mut [__m512d; MAX_BATCH_GROUPS],
    ) {
        match factor.indices.len() {
            1 => solve_lu_avx512_fixed::<1>(factor, work, solution),
            2 => solve_lu_avx512_fixed::<2>(factor, work, solution),
            3 => solve_lu_avx512_fixed::<3>(factor, work, solution),
            _ => solve_lu_avx512_generic(factor, work, solution),
        }
    }

    #[target_feature(enable = "avx512f")]
    fn solve_lu_avx512_fixed<const COUNT: usize>(
        factor: &PassiveFactor,
        work: &mut [__m512d; MAX_BATCH_GROUPS],
        solution: &mut [__m512d; MAX_BATCH_GROUPS],
    ) {
        for pivot in 0..COUNT {
            if factor.pivots[pivot] != pivot {
                work.swap(pivot, factor.pivots[pivot]);
            }
            let pivot_value = work[pivot];
            for row in (pivot + 1)..COUNT {
                let multiplier = _mm512_set1_pd(factor.lu[row * COUNT + pivot]);
                work[row] = _mm512_sub_pd(work[row], _mm512_mul_pd(multiplier, pivot_value));
            }
        }
        for row in (0..COUNT).rev() {
            let mut value = work[row];
            for column in (row + 1)..COUNT {
                let multiplier = _mm512_set1_pd(factor.lu[row * COUNT + column]);
                value = _mm512_sub_pd(value, _mm512_mul_pd(multiplier, solution[column]));
            }
            solution[row] = _mm512_div_pd(value, _mm512_set1_pd(factor.lu[row * COUNT + row]));
        }
    }

    #[target_feature(enable = "avx512f")]
    fn solve_lu_avx512_generic(
        factor: &PassiveFactor,
        work: &mut [__m512d; MAX_BATCH_GROUPS],
        solution: &mut [__m512d; MAX_BATCH_GROUPS],
    ) {
        let count = factor.indices.len();
        for pivot in 0..count {
            if factor.pivots[pivot] != pivot {
                work.swap(pivot, factor.pivots[pivot]);
            }
            let pivot_value = work[pivot];
            for row in (pivot + 1)..count {
                let multiplier = _mm512_set1_pd(factor.lu[row * count + pivot]);
                work[row] = _mm512_sub_pd(work[row], _mm512_mul_pd(multiplier, pivot_value));
            }
        }
        for row in (0..count).rev() {
            let mut value = work[row];
            for column in (row + 1)..count {
                let multiplier = _mm512_set1_pd(factor.lu[row * count + column]);
                value = _mm512_sub_pd(value, _mm512_mul_pd(multiplier, solution[column]));
            }
            solution[row] = _mm512_div_pd(value, _mm512_set1_pd(factor.lu[row * count + row]));
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use std::arch::aarch64::*;

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon")]
    pub unsafe fn solve_factor_neon(
        factor: &PassiveFactor,
        gram: &[f64],
        rhs: &[f64],
        samples: usize,
        groups: usize,
        sample_indices: &[usize],
        output: &mut [f64],
        fallback: &mut Vec<usize>,
    ) {
        let count = factor.indices.len();
        let chunks = sample_indices.len().div_ceil(2);
        let tolerance = vdupq_n_f64(ACTIVE_TOLERANCE);
        for chunk in 0..chunks {
            let first = chunk * 2;
            let lane_count = (sample_indices.len() - first).min(2);
            let lanes = [
                sample_indices[first],
                sample_indices[(first + 1).min(sample_indices.len() - 1)],
            ];
            let mut work = [vdupq_n_f64(0.0); MAX_BATCH_GROUPS];
            let mut solution = [vdupq_n_f64(0.0); MAX_BATCH_GROUPS];
            for row in 0..count {
                work[row] = vsetq_lane_f64(
                    rhs[factor.indices[row] * samples + lanes[1]],
                    vdupq_n_f64(rhs[factor.indices[row] * samples + lanes[0]]),
                    1,
                );
            }
            solve_lu_neon(factor, &mut work, &mut solution);

            let mut valid = vdupq_n_u64(u64::MAX);
            for value in solution.iter().take(count) {
                valid = vandq_u64(valid, vcgtq_f64(*value, tolerance));
            }
            for row in 0..groups {
                if factor.indices.contains(&row) {
                    continue;
                }
                let mut gradient = vsetq_lane_f64(
                    rhs[row * samples + lanes[1]],
                    vdupq_n_f64(rhs[row * samples + lanes[0]]),
                    1,
                );
                for (column, value) in solution.iter().take(count).enumerate() {
                    let coefficient = vdupq_n_f64(gram[row * groups + factor.indices[column]]);
                    gradient = vsubq_f64(gradient, vmulq_f64(coefficient, *value));
                }
                valid = vandq_u64(valid, vcleq_f64(gradient, tolerance));
            }

            let mut values = [0.0; 2];
            for (column, value) in solution.iter().take(count).enumerate() {
                unsafe { vst1q_f64(values.as_mut_ptr(), *value) };
                let base = factor.indices[column] * samples;
                output[base + lanes[0]] = values[0];
                if lane_count == 2 {
                    output[base + lanes[1]] = values[1];
                }
            }
            if vgetq_lane_u64(valid, 0) == 0 {
                fallback.push(lanes[0]);
            }
            if lane_count == 2 && vgetq_lane_u64(valid, 1) == 0 {
                fallback.push(lanes[1]);
            }
        }
    }

    #[target_feature(enable = "neon")]
    fn solve_lu_neon(
        factor: &PassiveFactor,
        work: &mut [float64x2_t; MAX_BATCH_GROUPS],
        solution: &mut [float64x2_t; MAX_BATCH_GROUPS],
    ) {
        match factor.indices.len() {
            1 => solve_lu_neon_fixed::<1>(factor, work, solution),
            2 => solve_lu_neon_fixed::<2>(factor, work, solution),
            3 => solve_lu_neon_fixed::<3>(factor, work, solution),
            _ => solve_lu_neon_generic(factor, work, solution),
        }
    }

    #[target_feature(enable = "neon")]
    fn solve_lu_neon_fixed<const COUNT: usize>(
        factor: &PassiveFactor,
        work: &mut [float64x2_t; MAX_BATCH_GROUPS],
        solution: &mut [float64x2_t; MAX_BATCH_GROUPS],
    ) {
        for pivot in 0..COUNT {
            if factor.pivots[pivot] != pivot {
                work.swap(pivot, factor.pivots[pivot]);
            }
            let pivot_value = work[pivot];
            for row in (pivot + 1)..COUNT {
                let multiplier = vdupq_n_f64(factor.lu[row * COUNT + pivot]);
                work[row] = vsubq_f64(work[row], vmulq_f64(multiplier, pivot_value));
            }
        }
        for row in (0..COUNT).rev() {
            let mut value = work[row];
            for column in (row + 1)..COUNT {
                let multiplier = vdupq_n_f64(factor.lu[row * COUNT + column]);
                value = vsubq_f64(value, vmulq_f64(multiplier, solution[column]));
            }
            solution[row] = vdivq_f64(value, vdupq_n_f64(factor.lu[row * COUNT + row]));
        }
    }

    #[target_feature(enable = "neon")]
    fn solve_lu_neon_generic(
        factor: &PassiveFactor,
        work: &mut [float64x2_t; MAX_BATCH_GROUPS],
        solution: &mut [float64x2_t; MAX_BATCH_GROUPS],
    ) {
        let count = factor.indices.len();
        for pivot in 0..count {
            if factor.pivots[pivot] != pivot {
                work.swap(pivot, factor.pivots[pivot]);
            }
            let pivot_value = work[pivot];
            for row in (pivot + 1)..count {
                let multiplier = vdupq_n_f64(factor.lu[row * count + pivot]);
                work[row] = vsubq_f64(work[row], vmulq_f64(multiplier, pivot_value));
            }
        }
        for row in (0..count).rev() {
            let mut value = work[row];
            for column in (row + 1)..count {
                let multiplier = vdupq_n_f64(factor.lu[row * count + column]);
                value = vsubq_f64(value, vmulq_f64(multiplier, solution[column]));
            }
            solution[row] = vdivq_f64(value, vdupq_n_f64(factor.lu[row * count + row]));
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
) -> u64 {
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
            None => return passive,
        }
    }
    out.iter().enumerate().fold(0_u64, |mask, (index, value)| {
        if *value > ACTIVE_TOLERANCE {
            mask | (1_u64 << index)
        } else {
            mask
        }
    })
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

    #[test]
    fn batched_warm_solves_match_individual_active_set_solves() {
        let groups = 4;
        let samples = 37;
        let gram = [
            4.0, 0.4, 0.2, 0.1, //
            0.4, 3.0, 0.3, 0.2, //
            0.2, 0.3, 2.5, 0.5, //
            0.1, 0.2, 0.5, 2.0,
        ];
        let mut initial_rhs = vec![0.0; groups * samples];
        for group in 0..groups {
            for sample in 0..samples {
                initial_rhs[group * samples + sample] =
                    ((sample * (group + 3) + group * 7) % 23) as f64 / 5.0 - 1.4;
            }
        }
        let zero_warm = vec![0.0; groups * samples];
        let mut warm = vec![0.0; groups * samples];
        let mut warm_masks = vec![0_u64; samples];
        let mut workspace = BatchWorkspace::new(groups, samples);
        solve_nnls_batch(
            &gram,
            &initial_rhs,
            &zero_warm,
            None,
            samples,
            groups,
            &mut warm,
            &mut warm_masks,
            &mut workspace,
        );

        let mut proposal_rhs = initial_rhs.clone();
        for group in 0..groups {
            for sample in 0..samples {
                let direction = if sample % 11 == group { -2.0 } else { 0.015 };
                proposal_rhs[group * samples + sample] += direction * (group + 1) as f64;
            }
        }
        let mut actual = vec![0.0; groups * samples];
        let mut actual_masks = vec![0_u64; samples];
        solve_nnls_batch(
            &gram,
            &proposal_rhs,
            &warm,
            Some(&warm_masks),
            samples,
            groups,
            &mut actual,
            &mut actual_masks,
            &mut workspace,
        );

        let mut cache = FactorCache::new(groups);
        let mut sample_rhs = vec![0.0; groups];
        let mut sample_warm = vec![0.0; groups];
        let mut expected = vec![0.0; groups];
        let mut candidate = vec![0.0; groups];
        for sample in 0..samples {
            for group in 0..groups {
                sample_rhs[group] = proposal_rhs[group * samples + sample];
                sample_warm[group] = warm[group * samples + sample];
            }
            let expected_mask = solve_nnls(
                &gram,
                &sample_rhs,
                &sample_warm,
                &mut expected,
                &mut candidate,
                &mut cache,
            );
            assert_eq!(actual_masks[sample], expected_mask, "sample={sample}");
            for group in 0..groups {
                assert!(
                    (actual[group * samples + sample] - expected[group]).abs() < 1.0e-12,
                    "sample={sample} group={group} actual={} expected={}",
                    actual[group * samples + sample],
                    expected[group]
                );
            }
        }
    }
}
