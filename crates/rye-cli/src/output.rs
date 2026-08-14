use crate::data::{Dataset, PopulationMap};
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn output_path(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path: OsString = prefix.as_os_str().to_owned();
    path.push(suffix);
    PathBuf::from(path)
}

fn format_probability(value: f64) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    let exponent = value.abs().log10().floor() as i32;
    let decimals = (14 - exponent).max(0) as usize;
    let mut formatted = format!("{value:.decimals$}");
    if formatted.contains('.') {
        while formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.ends_with('.') {
            formatted.pop();
        }
    }
    if formatted == "-0" {
        "0".to_owned()
    } else {
        formatted
    }
}

pub fn write_outputs(
    prefix: &Path,
    pcs: usize,
    dataset: &Dataset,
    mapping: &PopulationMap,
    estimates: &[f64],
) -> Result<(PathBuf, PathBuf), String> {
    let q_path = output_path(prefix, &format!("-{pcs}.{}.Q", mapping.groups.len()));
    let fam_path = output_path(prefix, &format!("-{pcs}.fam"));
    let mut q = BufWriter::new(
        File::create(&q_path)
            .map_err(|error| format!("failed to create {}: {error}", q_path.display()))?,
    );
    writeln!(q, "{}", mapping.groups.join("\t"))
        .map_err(|error| format!("failed to write {}: {error}", q_path.display()))?;
    for sample in 0..dataset.samples {
        write!(q, "{}", dataset.ids[sample])
            .map_err(|error| format!("failed to write {}: {error}", q_path.display()))?;
        for group in 0..mapping.groups.len() {
            write!(
                q,
                "\t{}",
                format_probability(estimates[group * dataset.samples + sample])
            )
            .map_err(|error| format!("failed to write {}: {error}", q_path.display()))?;
        }
        writeln!(q).map_err(|error| format!("failed to write {}: {error}", q_path.display()))?;
    }
    q.flush()
        .map_err(|error| format!("failed to flush {}: {error}", q_path.display()))?;

    let mut fam = BufWriter::new(
        File::create(&fam_path)
            .map_err(|error| format!("failed to create {}: {error}", fam_path.display()))?,
    );
    writeln!(fam, "population\tid")
        .map_err(|error| format!("failed to write {}: {error}", fam_path.display()))?;
    for sample in 0..dataset.samples {
        writeln!(
            fam,
            "{}\t{}\t{}",
            dataset.ids[sample], dataset.populations[sample], dataset.ids[sample]
        )
        .map_err(|error| format!("failed to write {}: {error}", fam_path.display()))?;
    }
    fam.flush()
        .map_err(|error| format!("failed to flush {}: {error}", fam_path.display()))?;
    Ok((q_path, fam_path))
}

#[cfg(test)]
mod tests {
    use super::format_probability;

    #[test]
    fn formats_like_r_write_table() {
        assert_eq!(format_probability(0.0), "0");
        assert_eq!(
            format_probability(0.145_442_823_004_484),
            "0.145442823004484"
        );
        assert_eq!(
            format_probability(0.960_483_845_299_909_6),
            "0.96048384529991"
        );
        assert_eq!(
            format_probability(0.013_063_198_152_082_6),
            "0.0130631981520826"
        );
    }
}
