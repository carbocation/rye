#!/usr/bin/env Rscript

scriptArgument = grep('^--file=', commandArgs(trailingOnly = FALSE), value = TRUE)[[1]]
repository = dirname(dirname(normalizePath(sub('^--file=', '', scriptArgument))))
source(file.path(repository, 'rye.R'))

stopifnot(rye.nativeSolver)

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
means = rye.populationMeans(
  referenceX, referenceFam, alpha, weight, referenceGroups = referenceGroups
)[groups, ]

emptyWarmStart = matrix(numeric(), nrow = 0, ncol = 0)
nativeCoefficients = .Call(
  'rye_nnls_batch', unname(referenceX), unname(means), weight, emptyWarmStart
)
warmCoefficients = .Call(
  'rye_nnls_batch', unname(referenceX), unname(means), weight, nativeCoefficients
)
stopifnot(max(abs(nativeCoefficients - warmCoefficients)) < 1e-10)

## Verify the NNLS KKT conditions without relying on another solver.
design = t(means)
weightedX = referenceX * rep(weight, each = nrow(referenceX))
for (row in seq_len(min(100, nrow(referenceX)))) {
  coefficients = nativeCoefficients[row, ]
  gradient = drop(crossprod(design, weightedX[row, ] - design %*% coefficients))
  stopifnot(all(gradient[coefficients == 0] < 1e-8))
  stopifnot(all(abs(gradient[coefficients > 0]) < 1e-8))
}

## The persistent native Gibbs loop must track the R-orchestrated loop under
## identical RNG state while avoiding per-proposal R allocations.
set.seed(123)
nativeOptimizer = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 500, sd = 0.01
)
rye.nativeOptimizer = FALSE
set.seed(123)
rOptimizer = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 500, sd = 0.01
)
stopifnot(abs(nativeOptimizer[[1]] - rOptimizer[[1]]) < 1e-9)
stopifnot(max(abs(nativeOptimizer[[2]] - rOptimizer[[2]])) < 1e-12)
stopifnot(max(abs(nativeOptimizer[[3]] - rOptimizer[[3]])) < 1e-12)
stopifnot(max(abs(nativeOptimizer[[4]] - rOptimizer[[4]])) < 1e-9)
stopifnot(max(abs(nativeOptimizer[[5]] - rOptimizer[[5]])) < 1e-8)
rye.nativeOptimizer = TRUE

## When nnls is installed, check numerical parity with Rye's compatibility path.
if (requireNamespace('nnls', quietly = TRUE)) {
  nativePredictions = rye.predict(referenceX, means, weight)
  rye.nativeSolver = FALSE
  suppressMessages(library(nnls))
  legacyPredictions = rye.predict(referenceX, means, weight)
  stopifnot(max(abs(nativePredictions - legacyPredictions)) < 1e-8)
}

message('Native solver verification passed')
