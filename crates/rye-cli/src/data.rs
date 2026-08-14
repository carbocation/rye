use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

pub struct Dataset {
    pub populations: Vec<String>,
    pub ids: Vec<String>,
    /// Column-major sample-by-feature matrix.
    pub x: Vec<f64>,
    pub samples: usize,
    pub features: usize,
}

pub struct PopulationMap {
    pub groups: Vec<String>,
    population_group: HashMap<String, usize>,
    group_index: HashMap<String, usize>,
}

pub struct ReferenceData {
    /// Column-major reference-sample-by-feature matrix.
    pub x: Vec<f64>,
    /// Column-major group-by-feature raw medians.
    pub raw_means: Vec<f64>,
    pub target_group: Vec<usize>,
    pub sample_weight: Vec<f64>,
    pub samples: usize,
}

fn parse_eigenvectors(contents: &str, pcs: usize, source: &Path) -> Result<Dataset, String> {
    let mut populations = Vec::new();
    let mut ids = Vec::new();
    let mut rows = Vec::new();
    let mut feature_count = None;
    let mut seen_ids = HashSet::new();
    for (line_index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Err(format!(
                "{}:{}: expected population, sample ID, and PCs",
                source.display(),
                line_index + 1
            ));
        }
        let current_features = fields.len() - 2;
        if let Some(expected) = feature_count {
            if current_features != expected {
                return Err(format!(
                    "{}:{}: expected {expected} PCs, found {current_features}",
                    source.display(),
                    line_index + 1
                ));
            }
        } else {
            feature_count = Some(current_features);
            if current_features < pcs {
                return Err(format!(
                    "{} contains {current_features} PCs but --pcs={pcs}",
                    source.display()
                ));
            }
        }
        if !seen_ids.insert(fields[1].to_owned()) {
            return Err(format!("duplicate sample ID: {}", fields[1]));
        }
        let mut values = Vec::with_capacity(current_features);
        for (feature, value) in fields[2..].iter().enumerate() {
            let parsed = value.parse::<f64>().map_err(|_| {
                format!(
                    "{}:{}: PC{} is not numeric: {value}",
                    source.display(),
                    line_index + 1,
                    feature + 1
                )
            })?;
            if !parsed.is_finite() {
                return Err(format!(
                    "{}:{}: PC{} is not finite",
                    source.display(),
                    line_index + 1,
                    feature + 1
                ));
            }
            values.push(parsed);
        }
        populations.push(fields[0].to_owned());
        ids.push(fields[1].to_owned());
        rows.push(values);
    }
    if rows.is_empty() {
        return Err(format!("{} contains no samples", source.display()));
    }
    let samples = rows.len();
    let mut x = vec![0.0; samples * pcs];
    for feature in 0..pcs {
        let minimum = rows
            .iter()
            .map(|row| row[feature])
            .fold(f64::INFINITY, f64::min);
        let maximum = rows
            .iter()
            .map(|row| row[feature])
            .fold(f64::NEG_INFINITY, f64::max);
        let range = maximum - minimum;
        if range == 0.0 {
            return Err(format!("PC{} has zero range", feature + 1));
        }
        for sample in 0..samples {
            x[feature * samples + sample] = (rows[sample][feature] - minimum) / range;
        }
    }
    Ok(Dataset {
        populations,
        ids,
        x,
        samples,
        features: pcs,
    })
}

pub fn read_eigenvectors(path: &Path, pcs: usize) -> Result<Dataset, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    parse_eigenvectors(&contents, pcs, path)
}

pub fn read_weights(path: &Path, pcs: usize) -> Result<Vec<f64>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut values = Vec::new();
    for (line_index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value = line
            .split_whitespace()
            .next()
            .expect("nonempty line")
            .parse::<f64>()
            .map_err(|_| {
                format!(
                    "{}:{}: eigenvalue is not numeric",
                    path.display(),
                    line_index + 1
                )
            })?;
        if !value.is_finite() {
            return Err(format!(
                "{}:{}: eigenvalue is not finite",
                path.display(),
                line_index + 1
            ));
        }
        values.push(value);
    }
    if values.len() < pcs {
        return Err(format!(
            "{} contains {} eigenvalues but --pcs={pcs}",
            path.display(),
            values.len()
        ));
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == 0.0 {
        return Err("maximum eigenvalue is zero".to_owned());
    }
    Ok(values[..pcs].iter().map(|value| value / maximum).collect())
}

pub fn read_population_map(path: &Path) -> Result<PopulationMap, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut lines = contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.trim().starts_with('#'));
    let (header_line, header) = lines
        .next()
        .ok_or_else(|| format!("{} contains no header", path.display()))?;
    let headers: Vec<&str> = header.split_whitespace().collect();
    let population_column = headers
        .iter()
        .position(|name| *name == "Pop")
        .ok_or_else(|| format!("{} header is missing Pop", path.display()))?;
    let group_column = headers
        .iter()
        .position(|name| *name == "Group")
        .ok_or_else(|| format!("{} header is missing Group", path.display()))?;
    let required_columns = population_column.max(group_column) + 1;
    let mut groups = Vec::new();
    let mut group_index = HashMap::new();
    let mut population_group = HashMap::new();
    for (line_index, line) in lines {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < required_columns {
            return Err(format!(
                "{}:{}: mapping row has too few columns (header is line {})",
                path.display(),
                line_index + 1,
                header_line + 1
            ));
        }
        let group = fields[group_column];
        let next_index = groups.len();
        let index = *group_index.entry(group.to_owned()).or_insert_with(|| {
            groups.push(group.to_owned());
            next_index
        });
        population_group
            .entry(fields[population_column].to_owned())
            .or_insert(index);
    }
    if groups.is_empty() {
        return Err(format!("{} contains no mappings", path.display()));
    }
    if groups.len() > 63 {
        return Err("at most 63 ancestry groups are supported".to_owned());
    }
    Ok(PopulationMap {
        groups,
        population_group,
        group_index,
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() & 1 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

impl PopulationMap {
    fn sample_group(&self, population: &str) -> Option<usize> {
        self.population_group
            .get(population)
            .or_else(|| self.group_index.get(population))
            .copied()
    }

    /// Preserve the historical R `aggregate()` row ordering used by alpha.
    pub fn alpha_for_group(&self) -> Vec<usize> {
        let mut alphabetical: Vec<usize> = (0..self.groups.len()).collect();
        alphabetical.sort_by(|&left, &right| self.groups[left].cmp(&self.groups[right]));
        let mut position = vec![0; self.groups.len()];
        for (alphabetical_position, group) in alphabetical.into_iter().enumerate() {
            position[group] = alphabetical_position;
        }
        position
    }

    pub fn prepare_reference(&self, dataset: &Dataset) -> Result<ReferenceData, String> {
        let mut source_rows = Vec::new();
        let mut target_group = Vec::new();
        let mut group_rows = vec![Vec::new(); self.groups.len()];
        for (sample, population) in dataset.populations.iter().enumerate() {
            if let Some(group) = self.sample_group(population) {
                let reference_row = source_rows.len();
                source_rows.push(sample);
                target_group.push(group);
                group_rows[group].push(reference_row);
            }
        }
        if source_rows.is_empty() {
            return Err("no samples match the population mapping".to_owned());
        }
        for (group, rows) in self.groups.iter().zip(&group_rows) {
            if rows.is_empty() {
                return Err(format!("ancestry group {group} has no reference samples"));
            }
        }
        let samples = source_rows.len();
        let mut x = vec![0.0; samples * dataset.features];
        for feature in 0..dataset.features {
            for (reference_row, &source_row) in source_rows.iter().enumerate() {
                x[feature * samples + reference_row] =
                    dataset.x[feature * dataset.samples + source_row];
            }
        }
        let mut raw_means = vec![0.0; self.groups.len() * dataset.features];
        for feature in 0..dataset.features {
            for (group, rows) in group_rows.iter().enumerate() {
                let values = rows.iter().map(|&row| x[feature * samples + row]).collect();
                raw_means[group + feature * self.groups.len()] = median(values);
            }
        }
        let sample_weight = target_group
            .iter()
            .map(|&group| 1.0 / (self.groups.len() * group_rows[group].len()) as f64)
            .collect();
        Ok(ReferenceData {
            x,
            raw_means,
            target_group,
            sample_weight,
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::parse_eigenvectors;
    use std::path::Path;

    #[test]
    fn parses_and_scales_eigenvectors() {
        let data = parse_eigenvectors(
            "#FID IID PC1 PC2\nA a -2 4\nB b 2 8\n",
            2,
            Path::new("fixture"),
        )
        .unwrap();
        assert_eq!(data.x, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(data.ids, ["a", "b"]);
    }
}
