mod args;
mod data;
mod output;

use args::{Action, Config};
use rye_core::{
    BatchNnlsInput, MathMode, OptimizeInput, lecuyer_seed, optimize_rounds, simd_level, solve_batch,
};
use std::env;
use std::process::ExitCode;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn generated_seed() -> i32 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let mixed = (nanos as u64) ^ ((nanos >> 64) as u64) ^ u64::from(std::process::id());
    let seed = (mixed ^ mixed.rotate_left(29)) as i32;
    if seed == i32::MIN { i32::MAX } else { seed }
}

fn deterministic_math() -> bool {
    env::var("RYE_DETERMINISTIC_MATH")
        .is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

fn kernel_name() -> &'static str {
    match simd_level() {
        1 => "NEON",
        2 => "AVX2",
        3 => "AVX-512",
        _ => "scalar",
    }
}

fn run(config: Config) -> Result<(), String> {
    let started = Instant::now();
    println!("Reading eigenvectors from {}", config.eigenvec.display());
    let dataset = data::read_eigenvectors(&config.eigenvec, config.pcs)?;
    println!("Reading eigenvalues from {}", config.eigenval.display());
    let initial_weight = data::read_weights(&config.eigenval, config.pcs)?;
    println!(
        "Reading population mapping from {}",
        config.pop2group.display()
    );
    let mapping = data::read_population_map(&config.pop2group)?;
    let reference = mapping.prepare_reference(&dataset)?;
    let groups = mapping.groups.len();
    let initial_alpha = vec![0.001; groups];
    let alpha_for_group = mapping.alpha_for_group();
    let seed = config.seed.unwrap_or_else(generated_seed);
    let random_seed = lecuyer_seed(seed).map_err(|error| error.to_string())?;
    let math_mode = if deterministic_math() {
        MathMode::Deterministic
    } else {
        MathMode::Platform
    };
    println!(
        "Optimizing with {} worker(s), {} attempt(s), {} kernel, seed {}",
        config.threads,
        config.attempts,
        kernel_name(),
        seed
    );
    let optimized = optimize_rounds(
        OptimizeInput {
            x: &reference.x,
            raw_means: &reference.raw_means,
            alpha: &initial_alpha,
            weight: &initial_weight,
            alpha_for_group: &alpha_for_group,
            target_group: &reference.target_group,
            sample_weight: &reference.sample_weight,
            samples: reference.samples,
            features: dataset.features,
            groups,
            iterations: config.iterations,
            rounds: config.rounds,
            attempts: config.attempts,
            threads: config.threads,
            start_sd: 0.01,
            end_sd: 0.005,
            optimize_alpha: true,
            optimize_weight: true,
            math_mode,
            random_seed: &random_seed,
        },
        |progress| {
            println!(
                "Round {}/{} Mean error: {:.6}, Best error: {:.6}",
                progress.round, progress.rounds, progress.mean_error, progress.best_error
            );
        },
    )
    .map_err(|error| error.to_string())?;

    let mut means = vec![0.0; groups * dataset.features];
    for feature in 0..dataset.features {
        for group in 0..groups {
            means[group + feature * groups] =
                optimized.best.basis[group * dataset.features + feature];
        }
    }
    let mut estimates = solve_batch(BatchNnlsInput {
        x: &dataset.x,
        means: &means,
        weights: &optimized.best.weight,
        warm: &[],
        samples: dataset.samples,
        features: dataset.features,
        groups,
    })
    .map_err(|error| error.to_string())?;
    // `rye.predict()` normalizes once, and the historical top-level R workflow
    // normalizes its result a second time before writing output.
    for _ in 0..2 {
        for sample in 0..dataset.samples {
            let total: f64 = (0..groups)
                .map(|group| estimates[group * dataset.samples + sample])
                .sum();
            if !total.is_finite() || total <= 0.0 {
                return Err(format!(
                    "sample {} produced invalid ancestry coefficients",
                    dataset.ids[sample]
                ));
            }
            for group in 0..groups {
                estimates[group * dataset.samples + sample] /= total;
            }
        }
    }
    let (q_path, fam_path) =
        output::write_outputs(&config.output, config.pcs, &dataset, &mapping, &estimates)?;
    println!("Wrote {}", q_path.display());
    println!("Wrote {}", fam_path.display());
    println!(
        "Completed in {:.3} seconds",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    match args::parse(env::args().skip(1)) {
        Ok(Action::Help) => {
            print!("{}", args::HELP);
            ExitCode::SUCCESS
        }
        Ok(Action::Version) => {
            println!("rye {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Action::Run(config)) => match run(config) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("error: {error}\n\n{}", args::HELP);
            ExitCode::FAILURE
        }
    }
}
