pub type AxpyKernel = fn(&mut [f64], &[f64], f64);
pub type PredictionErrorKernel = fn(&[f64], &[usize], &[f64], usize, usize) -> f64;

#[inline]
pub fn scalar_axpy(output: &mut [f64], input: &[f64], scale: f64) {
    for (destination, &source) in output.iter_mut().zip(input) {
        *destination += source * scale;
    }
}

pub fn scalar_prediction_error(
    coefficients: &[f64],
    target_group: &[usize],
    sample_weight: &[f64],
    samples: usize,
    groups: usize,
) -> f64 {
    scalar_prediction_error_range(
        coefficients,
        target_group,
        sample_weight,
        samples,
        groups,
        0,
        0.0,
    )
}

fn scalar_prediction_error_range(
    coefficients: &[f64],
    target_group: &[usize],
    sample_weight: &[f64],
    samples: usize,
    groups: usize,
    start: usize,
    mut error: f64,
) -> f64 {
    for sample in start..samples {
        let mut total = 0.0;
        for group in 0..groups {
            total += coefficients[group * samples + sample];
        }
        let target = coefficients[target_group[sample] * samples + sample] / total;
        error += sample_weight[sample] * (2.0 * (1.0 - target) / groups as f64);
    }
    error
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_axpy_impl(output: &mut [f64], input: &[f64], scale: f64) {
        let scale_vector = _mm256_set1_pd(scale);
        let chunks = output.len() / 4;
        for chunk in 0..chunks {
            let index = chunk * 4;
            let source = unsafe { _mm256_loadu_pd(input.as_ptr().add(index)) };
            let destination = unsafe { _mm256_loadu_pd(output.as_ptr().add(index)) };
            let product = _mm256_mul_pd(source, scale_vector);
            unsafe {
                _mm256_storeu_pd(
                    output.as_mut_ptr().add(index),
                    _mm256_add_pd(destination, product),
                );
            }
        }
        super::scalar_axpy(&mut output[chunks * 4..], &input[chunks * 4..], scale);
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_axpy_impl(output: &mut [f64], input: &[f64], scale: f64) {
        let scale_vector = _mm512_set1_pd(scale);
        let chunks = output.len() / 8;
        for chunk in 0..chunks {
            let index = chunk * 8;
            let source = unsafe { _mm512_loadu_pd(input.as_ptr().add(index)) };
            let destination = unsafe { _mm512_loadu_pd(output.as_ptr().add(index)) };
            let product = _mm512_mul_pd(source, scale_vector);
            unsafe {
                _mm512_storeu_pd(
                    output.as_mut_ptr().add(index),
                    _mm512_add_pd(destination, product),
                );
            }
        }
        super::scalar_axpy(&mut output[chunks * 8..], &input[chunks * 8..], scale);
    }

    pub fn avx2_axpy(output: &mut [f64], input: &[f64], scale: f64) {
        unsafe { avx2_axpy_impl(output, input, scale) }
    }

    pub fn avx512_axpy(output: &mut [f64], input: &[f64], scale: f64) {
        unsafe { avx512_axpy_impl(output, input, scale) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_prediction_error_impl(
        coefficients: &[f64],
        target_group: &[usize],
        sample_weight: &[f64],
        samples: usize,
        groups: usize,
    ) -> f64 {
        let chunks = samples / 4;
        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let group_count = _mm256_set1_pd(groups as f64);
        let mut error = 0.0;
        let mut contributions = [0.0; 4];
        for chunk in 0..chunks {
            let sample = chunk * 4;
            let mut total = _mm256_setzero_pd();
            for group in 0..groups {
                let values =
                    unsafe { _mm256_loadu_pd(coefficients.as_ptr().add(group * samples + sample)) };
                total = _mm256_add_pd(total, values);
            }
            let targets = _mm256_set_pd(
                coefficients[target_group[sample + 3] * samples + sample + 3],
                coefficients[target_group[sample + 2] * samples + sample + 2],
                coefficients[target_group[sample + 1] * samples + sample + 1],
                coefficients[target_group[sample] * samples + sample],
            );
            let target_fraction = _mm256_div_pd(targets, total);
            let scaled = _mm256_div_pd(
                _mm256_mul_pd(two, _mm256_sub_pd(one, target_fraction)),
                group_count,
            );
            let weights = unsafe { _mm256_loadu_pd(sample_weight.as_ptr().add(sample)) };
            unsafe { _mm256_storeu_pd(contributions.as_mut_ptr(), _mm256_mul_pd(weights, scaled)) };
            for contribution in contributions {
                error += contribution;
            }
        }
        super::scalar_prediction_error_range(
            coefficients,
            target_group,
            sample_weight,
            samples,
            groups,
            chunks * 4,
            error,
        )
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_prediction_error_impl(
        coefficients: &[f64],
        target_group: &[usize],
        sample_weight: &[f64],
        samples: usize,
        groups: usize,
    ) -> f64 {
        let chunks = samples / 8;
        let one = _mm512_set1_pd(1.0);
        let two = _mm512_set1_pd(2.0);
        let group_count = _mm512_set1_pd(groups as f64);
        let mut error = 0.0;
        let mut contributions = [0.0; 8];
        for chunk in 0..chunks {
            let sample = chunk * 8;
            let mut total = _mm512_setzero_pd();
            for group in 0..groups {
                let values =
                    unsafe { _mm512_loadu_pd(coefficients.as_ptr().add(group * samples + sample)) };
                total = _mm512_add_pd(total, values);
            }
            let targets = _mm512_set_pd(
                coefficients[target_group[sample + 7] * samples + sample + 7],
                coefficients[target_group[sample + 6] * samples + sample + 6],
                coefficients[target_group[sample + 5] * samples + sample + 5],
                coefficients[target_group[sample + 4] * samples + sample + 4],
                coefficients[target_group[sample + 3] * samples + sample + 3],
                coefficients[target_group[sample + 2] * samples + sample + 2],
                coefficients[target_group[sample + 1] * samples + sample + 1],
                coefficients[target_group[sample] * samples + sample],
            );
            let target_fraction = _mm512_div_pd(targets, total);
            let scaled = _mm512_div_pd(
                _mm512_mul_pd(two, _mm512_sub_pd(one, target_fraction)),
                group_count,
            );
            let weights = unsafe { _mm512_loadu_pd(sample_weight.as_ptr().add(sample)) };
            unsafe { _mm512_storeu_pd(contributions.as_mut_ptr(), _mm512_mul_pd(weights, scaled)) };
            for contribution in contributions {
                error += contribution;
            }
        }
        super::scalar_prediction_error_range(
            coefficients,
            target_group,
            sample_weight,
            samples,
            groups,
            chunks * 8,
            error,
        )
    }

    pub fn avx2_prediction_error(
        coefficients: &[f64],
        target_group: &[usize],
        sample_weight: &[f64],
        samples: usize,
        groups: usize,
    ) -> f64 {
        unsafe {
            avx2_prediction_error_impl(coefficients, target_group, sample_weight, samples, groups)
        }
    }

    pub fn avx512_prediction_error(
        coefficients: &[f64],
        target_group: &[usize],
        sample_weight: &[f64],
        samples: usize,
        groups: usize,
    ) -> f64 {
        unsafe {
            avx512_prediction_error_impl(coefficients, target_group, sample_weight, samples, groups)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    unsafe fn neon_axpy_impl(output: &mut [f64], input: &[f64], scale: f64) {
        let chunks = output.len() / 2;
        for chunk in 0..chunks {
            let index = chunk * 2;
            let source = unsafe { vld1q_f64(input.as_ptr().add(index)) };
            let destination = unsafe { vld1q_f64(output.as_ptr().add(index)) };
            let product = vmulq_n_f64(source, scale);
            unsafe {
                vst1q_f64(
                    output.as_mut_ptr().add(index),
                    vaddq_f64(destination, product),
                );
            }
        }
        super::scalar_axpy(&mut output[chunks * 2..], &input[chunks * 2..], scale);
    }

    pub fn neon_axpy(output: &mut [f64], input: &[f64], scale: f64) {
        unsafe { neon_axpy_impl(output, input, scale) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_prediction_error_impl(
        coefficients: &[f64],
        target_group: &[usize],
        sample_weight: &[f64],
        samples: usize,
        groups: usize,
    ) -> f64 {
        let chunks = samples / 2;
        let one = vdupq_n_f64(1.0);
        let two = vdupq_n_f64(2.0);
        let group_count = vdupq_n_f64(groups as f64);
        let mut error = 0.0;
        let mut contributions = [0.0; 2];
        for chunk in 0..chunks {
            let sample = chunk * 2;
            let mut total = vdupq_n_f64(0.0);
            for group in 0..groups {
                let values =
                    unsafe { vld1q_f64(coefficients.as_ptr().add(group * samples + sample)) };
                total = vaddq_f64(total, values);
            }
            let targets = vsetq_lane_f64(
                coefficients[target_group[sample + 1] * samples + sample + 1],
                vdupq_n_f64(coefficients[target_group[sample] * samples + sample]),
                1,
            );
            let target_fraction = vdivq_f64(targets, total);
            let scaled = vdivq_f64(vmulq_f64(two, vsubq_f64(one, target_fraction)), group_count);
            let weights = unsafe { vld1q_f64(sample_weight.as_ptr().add(sample)) };
            unsafe { vst1q_f64(contributions.as_mut_ptr(), vmulq_f64(weights, scaled)) };
            error += contributions[0];
            error += contributions[1];
        }
        super::scalar_prediction_error_range(
            coefficients,
            target_group,
            sample_weight,
            samples,
            groups,
            chunks * 2,
            error,
        )
    }

    pub fn neon_prediction_error(
        coefficients: &[f64],
        target_group: &[usize],
        sample_weight: &[f64],
        samples: usize,
        groups: usize,
    ) -> f64 {
        unsafe {
            neon_prediction_error_impl(coefficients, target_group, sample_weight, samples, groups)
        }
    }
}

pub fn select_axpy() -> AxpyKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return x86::avx512_axpy;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86::avx2_axpy;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return arm::neon_axpy;
    }
    #[allow(unreachable_code)]
    scalar_axpy
}

pub fn select_prediction_error() -> PredictionErrorKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return x86::avx512_prediction_error;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86::avx2_prediction_error;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return arm::neon_prediction_error;
    }
    #[allow(unreachable_code)]
    scalar_prediction_error
}

pub fn kernel_level() -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return 3;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return 2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return 1;
    }
    #[allow(unreachable_code)]
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_kernel_matches_scalar() {
        let input: Vec<f64> = (0..37).map(|value| value as f64 / 13.0).collect();
        let mut expected: Vec<f64> = (0..37).map(|value| value as f64 / 7.0).collect();
        let mut actual = expected.clone();
        scalar_axpy(&mut expected, &input, 0.375);
        select_axpy()(&mut actual, &input, 0.375);
        assert_eq!(actual, expected);
    }

    #[test]
    fn selected_prediction_error_matches_scalar_order() {
        let samples = 37;
        let groups = 7;
        let coefficients: Vec<f64> = (0..groups)
            .flat_map(|group| {
                (0..samples).map(move |sample| {
                    if (sample + group * 3) % 11 == 0 {
                        0.0
                    } else {
                        ((sample * 5 + group * 13) % 29 + 1) as f64 / 31.0
                    }
                })
            })
            .collect();
        let target_group: Vec<usize> = (0..samples).map(|sample| sample % groups).collect();
        let sample_weight: Vec<f64> = (0..samples)
            .map(|sample| (sample % 9 + 1) as f64 / 17.0)
            .collect();
        let expected = scalar_prediction_error(
            &coefficients,
            &target_group,
            &sample_weight,
            samples,
            groups,
        );
        let actual = select_prediction_error()(
            &coefficients,
            &target_group,
            &sample_weight,
            samples,
            groups,
        );
        assert_eq!(actual, expected);
    }
}
