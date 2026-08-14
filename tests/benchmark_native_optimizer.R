#!/usr/bin/env Rscript

args = commandArgs(trailingOnly = TRUE)
repository = if (length(args) >= 1) normalizePath(args[[1]]) else normalizePath('.')
iterations = if (length(args) >= 2) as.integer(args[[2]]) else 2000L
repeats = if (length(args) >= 3) as.integer(args[[3]]) else 7L
source(file.path(repository, 'rye.R'))

stopifnot(rye.nativeOptimizer)
hasPureRustRng = exists('rye.nativeDeterministicMath', inherits = FALSE)

fullPCA = read.table(file.path(repository, 'examples', 'example.eigenvec'), header = FALSE)
rownames(fullPCA) = fullPCA[ , 2]
fam = as.matrix(fullPCA[ , c(1, 2)])
colnames(fam) = c('population', 'id')
rownames(fam) = fam[ , 'id']
X = rye.scale(as.matrix(fullPCA[ , 3:22]))

pop2group = read.table(
  file.path(repository, 'examples', 'pop2group.txt'),
  header = TRUE, stringsAsFactors = FALSE
)
populationGroups = setNames(pop2group$Group, pop2group$Pop)
isReference = fam[ , 'population'] %in% names(populationGroups)
fam[isReference, 'population'] = populationGroups[fam[isReference, 'population']]
groups = unique(populationGroups)
referenceGroups = setNames(groups, groups)
referenceFam = fam[fam[ , 'population'] %in% groups, , drop = FALSE]
referenceX = X[rownames(referenceFam), , drop = FALSE]
alpha = setNames(rep(0.001, length(groups)), groups)
weight = scan(file.path(repository, 'examples', 'example.eigenval'), quiet = TRUE)[1:20]
weight = weight / max(weight)

benchmarkMode = function(label, deterministic = FALSE) {
  if (hasPureRustRng) rye.nativeDeterministicMath <<- deterministic
  set.seed(123)
  invisible(rye.gibbs(
    referenceX, referenceFam, referenceGroups,
    alpha, TRUE, weight, TRUE, iterations = 50, sd = 0.01
  ))
  elapsed = numeric(repeats)
  for (index in seq_len(repeats)) {
    set.seed(123)
    elapsed[[index]] = system.time(invisible(rye.gibbs(
      referenceX, referenceFam, referenceGroups,
      alpha, TRUE, weight, TRUE, iterations = iterations, sd = 0.01
    )))[['elapsed']]
  }
  cat(sprintf(
    '%s: median %.6fs, min %.6fs, %.3f us/proposal\n',
    label, median(elapsed), min(elapsed), 1e6 * median(elapsed) / iterations
  ))
}

if (hasPureRustRng) {
  benchmarkMode('pure Rust RNG, platform math', FALSE)
  benchmarkMode('pure Rust RNG, deterministic math', TRUE)
} else {
  benchmarkMode('R RNG/math callbacks')
}
