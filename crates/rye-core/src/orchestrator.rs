use super::{CoreError, GibbsInput, GibbsResult, MathMode, lecuyer_stream_seeds, optimize_gibbs};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;
use std::thread;

pub struct OptimizeInput<'a> {
    /// Column-major sample-by-feature matrix containing reference samples.
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
    pub rounds: usize,
    pub attempts: usize,
    pub threads: usize,
    pub start_sd: f64,
    pub end_sd: f64,
    pub optimize_alpha: bool,
    pub optimize_weight: bool,
    pub math_mode: MathMode,
    /// Serialized L'Ecuyer-CMRG state for logical round 1, attempt 1.
    pub random_seed: &'a [i32],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundProgress {
    pub round: usize,
    pub rounds: usize,
    pub mean_error: f64,
    pub best_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizeResult {
    pub best: GibbsResult,
    pub rounds_completed: usize,
}

struct AttemptJob {
    index: usize,
    alpha: Vec<f64>,
    weight: Vec<f64>,
    proposal_sd: f64,
    random_seed: Vec<i32>,
}

pub fn optimize_rounds(
    input: OptimizeInput<'_>,
    mut report_progress: impl FnMut(RoundProgress),
) -> Result<OptimizeResult, CoreError> {
    if input.rounds == 0 || input.attempts == 0 || input.threads == 0 {
        return Err(CoreError::InvalidDimensions(
            "rounds, attempts, and threads must be nonzero",
        ));
    }
    super::validate_gibbs(&GibbsInput {
        x: input.x,
        raw_means: input.raw_means,
        alpha: input.alpha,
        weight: input.weight,
        alpha_for_group: input.alpha_for_group,
        target_group: input.target_group,
        sample_weight: input.sample_weight,
        samples: input.samples,
        features: input.features,
        groups: input.groups,
        iterations: input.iterations,
        proposal_sd: input.start_sd,
        optimize_alpha: input.optimize_alpha,
        optimize_weight: input.optimize_weight,
        math_mode: input.math_mode,
        random_seed: input.random_seed,
    })?;
    let stream_count =
        input
            .rounds
            .checked_mul(input.attempts)
            .ok_or(CoreError::InvalidDimensions(
                "rounds multiplied by attempts exceeds addressable memory",
            ))?;
    let streams = lecuyer_stream_seeds(input.random_seed, stream_count)?;
    let worker_count = input.threads.min(input.attempts);
    let x = input.x;
    let raw_means = input.raw_means;
    let alpha_for_group = input.alpha_for_group;
    let target_group = input.target_group;
    let sample_weight = input.sample_weight;
    let samples = input.samples;
    let features = input.features;
    let groups = input.groups;
    let iterations = input.iterations;
    let optimize_alpha = input.optimize_alpha;
    let optimize_weight = input.optimize_weight;
    let math_mode = input.math_mode;
    let mut final_result = None;

    thread::scope(|scope| -> Result<(), CoreError> {
        let (result_sender, result_receiver) = mpsc::channel();
        let mut job_senders = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let (job_sender, job_receiver) = mpsc::channel::<AttemptJob>();
            let result_sender = result_sender.clone();
            scope.spawn(move || {
                while let Ok(job) = job_receiver.recv() {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        optimize_gibbs(GibbsInput {
                            x,
                            raw_means,
                            alpha: &job.alpha,
                            weight: &job.weight,
                            alpha_for_group,
                            target_group,
                            sample_weight,
                            samples,
                            features,
                            groups,
                            iterations,
                            proposal_sd: job.proposal_sd,
                            optimize_alpha,
                            optimize_weight,
                            math_mode,
                            random_seed: &job.random_seed,
                        })
                    }))
                    .unwrap_or(Err(CoreError::WorkerPanicked));
                    if result_sender.send((job.index, result)).is_err() {
                        break;
                    }
                }
            });
            job_senders.push(job_sender);
        }
        drop(result_sender);

        let mut alpha = input.alpha.to_vec();
        let mut weight = input.weight.to_vec();
        let mut best_errors = Vec::with_capacity(input.rounds);
        for round_index in 0..input.rounds {
            let round_number = round_index + 1;
            let proposal_sd = if input.rounds == 1 {
                input.start_sd
            } else {
                input.start_sd
                    - (input.start_sd - input.end_sd) * (round_number as f64).ln()
                        / (input.rounds as f64).ln()
            };
            for attempt_index in 0..input.attempts {
                let stream_index = round_index * input.attempts + attempt_index;
                let seed_offset = stream_index * super::LECUYER_RANDOM_SEED_LEN;
                job_senders[attempt_index % worker_count]
                    .send(AttemptJob {
                        index: attempt_index,
                        alpha: alpha.clone(),
                        weight: weight.clone(),
                        proposal_sd,
                        random_seed: streams
                            [seed_offset..seed_offset + super::LECUYER_RANDOM_SEED_LEN]
                            .to_vec(),
                    })
                    .map_err(|_| CoreError::WorkerDisconnected)?;
            }

            let mut results: Vec<Option<GibbsResult>> = std::iter::repeat_with(|| None)
                .take(input.attempts)
                .collect();
            for _ in 0..input.attempts {
                let (attempt_index, result) = result_receiver
                    .recv()
                    .map_err(|_| CoreError::WorkerDisconnected)?;
                results[attempt_index] = Some(result?);
            }
            let mean_error = results
                .iter()
                .map(|result| result.as_ref().expect("all attempts returned").best_error)
                .sum::<f64>()
                / input.attempts as f64;
            let best_index = results
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    left.as_ref()
                        .expect("all attempts returned")
                        .best_error
                        .total_cmp(&right.as_ref().expect("all attempts returned").best_error)
                })
                .map(|(index, _)| index)
                .expect("attempts is nonzero");
            let best = results[best_index].take().expect("best attempt returned");
            alpha.clone_from(&best.alpha);
            weight.clone_from(&best.weight);
            best_errors.push(best.best_error);
            report_progress(RoundProgress {
                round: round_number,
                rounds: input.rounds,
                mean_error,
                best_error: best.best_error,
            });
            final_result = Some(OptimizeResult {
                best,
                rounds_completed: round_number,
            });

            if round_number > 5 {
                let recent = &best_errors[round_index - 5..=round_index];
                let minimum = recent.iter().copied().fold(f64::INFINITY, f64::min);
                let maximum = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if maximum - minimum <= 0.000_025 {
                    break;
                }
            }
        }
        drop(job_senders);
        Ok(())
    })?;

    final_result.ok_or(CoreError::InvalidDimensions("optimizer produced no rounds"))
}

#[cfg(test)]
mod tests {
    use super::{OptimizeInput, optimize_rounds};
    use crate::{MathMode, lecuyer_seed};

    fn run(threads: usize) -> super::OptimizeResult {
        let x = [0.9, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.9];
        let raw_means = [0.85, 0.15, 0.15, 0.85];
        let alpha = [0.001, 0.001];
        let weight = [1.0, 0.5];
        let alpha_for_group = [0, 1];
        let target_group = [0, 0, 1, 1];
        let sample_weight = [0.25; 4];
        let random_seed = lecuyer_seed(2026).unwrap();
        optimize_rounds(
            OptimizeInput {
                x: &x,
                raw_means: &raw_means,
                alpha: &alpha,
                weight: &weight,
                alpha_for_group: &alpha_for_group,
                target_group: &target_group,
                sample_weight: &sample_weight,
                samples: 4,
                features: 2,
                groups: 2,
                iterations: 25,
                rounds: 3,
                attempts: 4,
                threads,
                start_sd: 0.01,
                end_sd: 0.005,
                optimize_alpha: true,
                optimize_weight: true,
                math_mode: MathMode::Platform,
                random_seed: &random_seed,
            },
            |_| {},
        )
        .unwrap()
    }

    #[test]
    fn optimizer_is_independent_of_worker_count() {
        assert_eq!(run(1), run(2));
        assert_eq!(run(2), run(4));
    }
}
