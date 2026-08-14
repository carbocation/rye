pub type AxpyKernel = fn(&mut [f64], &[f64], f64);

#[inline]
pub fn scalar_axpy(output: &mut [f64], input: &[f64], scale: f64) {
    for (destination, &source) in output.iter_mut().zip(input) {
        *destination += source * scale;
    }
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
}
