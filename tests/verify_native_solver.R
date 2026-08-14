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

## When nnls is installed, check numerical parity with Rye's compatibility path.
if (requireNamespace('nnls', quietly = TRUE)) {
  nativePredictions = rye.predict(referenceX, means, weight)
  rye.nativeSolver = FALSE
  suppressMessages(library(nnls))
  legacyPredictions = rye.predict(referenceX, means, weight)
  stopifnot(max(abs(nativePredictions - legacyPredictions)) < 1e-8)
}

message('Native solver verification passed')
