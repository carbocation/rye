use std::path::PathBuf;

pub const HELP: &str = r#"Rye - Rapid ancestry estimation

Usage: rye [options]

Required:
    --eigenvec=<FILE>      PCA eigenvector file
    --eigenval=<FILE>      PCA eigenvalue file
    --pop2group=<FILE>     Population-to-group mapping file

Options:
    --output=<PREFIX>      Output prefix (default: output)
    --threads=<N>          Native attempt workers (default: 4)
    --pcs=<N>              Principal components to use (default: 20)
    --rounds=<N>           Optimization rounds (default: 200)
    --iter=<N>             Proposals per attempt (default: 100)
    --attempts=<N>         Attempts per round (default: 4)
    --seed=<INTEGER>       Reproducible 32-bit seed
    -h, --help             Show this help
    --version              Show the version
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub eigenvec: PathBuf,
    pub eigenval: PathBuf,
    pub pop2group: PathBuf,
    pub output: PathBuf,
    pub threads: usize,
    pub pcs: usize,
    pub rounds: usize,
    pub iterations: usize,
    pub attempts: usize,
    pub seed: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Run(Config),
    Help,
    Version,
}

fn parse_positive(value: &str, option: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{option} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{option} must be a positive integer"));
    }
    Ok(parsed)
}

pub fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Action, String> {
    let mut eigenvec = None;
    let mut eigenval = None;
    let mut pop2group = None;
    let mut output = PathBuf::from("output");
    let mut threads = 4;
    let mut pcs = 20;
    let mut rounds = 200;
    let mut iterations = 100;
    let mut attempts = 4;
    let mut seed = None;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        if argument == "-h" || argument == "--help" {
            return Ok(Action::Help);
        }
        if argument == "--version" {
            return Ok(Action::Version);
        }
        if !argument.starts_with("--") {
            return Err(format!("unexpected positional argument: {argument}"));
        }
        let (option, inline_value) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(name, value)| {
                (name, Some(value.to_owned()))
            });
        let value = match inline_value {
            Some(value) if !value.is_empty() => value,
            Some(_) => return Err(format!("{option} requires a value")),
            None => arguments
                .next()
                .ok_or_else(|| format!("{option} requires a value"))?,
        };
        match option {
            "--eigenvec" => eigenvec = Some(PathBuf::from(value)),
            "--eigenval" => eigenval = Some(PathBuf::from(value)),
            "--pop2group" => pop2group = Some(PathBuf::from(value)),
            "--output" => output = PathBuf::from(value),
            "--threads" => threads = parse_positive(&value, option)?,
            "--pcs" => pcs = parse_positive(&value, option)?,
            "--rounds" => rounds = parse_positive(&value, option)?,
            "--iter" => iterations = parse_positive(&value, option)?,
            "--attempts" => attempts = parse_positive(&value, option)?,
            "--seed" => {
                let parsed = value
                    .parse::<i32>()
                    .map_err(|_| "--seed must be a non-missing 32-bit integer".to_owned())?;
                if parsed == i32::MIN {
                    return Err("--seed cannot be R's NA integer value".to_owned());
                }
                seed = Some(parsed);
            }
            _ => return Err(format!("unknown option: {option}")),
        }
    }
    Ok(Action::Run(Config {
        eigenvec: eigenvec.ok_or_else(|| "--eigenvec is required".to_owned())?,
        eigenval: eigenval.ok_or_else(|| "--eigenval is required".to_owned())?,
        pop2group: pop2group.ok_or_else(|| "--pop2group is required".to_owned())?,
        output,
        threads,
        pcs,
        rounds,
        iterations,
        attempts,
        seed,
    }))
}

#[cfg(test)]
mod tests {
    use super::{Action, parse};

    #[test]
    fn parses_compatible_option_forms() {
        let Action::Run(config) = parse([
            "--eigenvec=x.eigenvec".to_owned(),
            "--eigenval".to_owned(),
            "x.eigenval".to_owned(),
            "--pop2group=groups.tsv".to_owned(),
            "--threads=8".to_owned(),
            "--seed=-42".to_owned(),
        ])
        .unwrap() else {
            panic!("expected runnable configuration");
        };
        assert_eq!(config.threads, 8);
        assert_eq!(config.seed, Some(-42));
        assert_eq!(config.pcs, 20);
    }

    #[test]
    fn rejects_zero_work() {
        let error = parse([
            "--eigenvec=x".to_owned(),
            "--eigenval=y".to_owned(),
            "--pop2group=z".to_owned(),
            "--threads=0".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("positive integer"));
    }
}
