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
nativeSeed = .Random.seed
rye.nativeOptimizer = FALSE
set.seed(123)
rOptimizer = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 500, sd = 0.01
)
rSeed = .Random.seed
stopifnot(abs(nativeOptimizer[[1]] - rOptimizer[[1]]) < 1e-9)
stopifnot(max(abs(nativeOptimizer[[2]] - rOptimizer[[2]])) < 1e-12)
stopifnot(max(abs(nativeOptimizer[[3]] - rOptimizer[[3]])) < 1e-12)
stopifnot(max(abs(nativeOptimizer[[4]] - rOptimizer[[4]])) < 1e-9)
stopifnot(max(abs(nativeOptimizer[[5]] - rOptimizer[[5]])) < 1e-8)
stopifnot(identical(nativeSeed, rSeed))
rye.nativeOptimizer = TRUE

## Conditional draw paths must consume the same state in every optimization
## mode, including the acceptance-only path.
optimizationModes = list(
  c(TRUE, FALSE), c(FALSE, TRUE), c(FALSE, FALSE)
)
for (modeIndex in seq_along(optimizationModes)) {
  mode = optimizationModes[[modeIndex]]
  set.seed(1000 + modeIndex)
  modeNative = rye.gibbs(
    referenceX, referenceFam, referenceGroups,
    alpha, mode[[1]], weight, mode[[2]], iterations = 75, sd = 0.01
  )
  modeNativeSeed = .Random.seed
  rye.nativeOptimizer = FALSE
  set.seed(1000 + modeIndex)
  modeR = rye.gibbs(
    referenceX, referenceFam, referenceGroups,
    alpha, mode[[1]], weight, mode[[2]], iterations = 75, sd = 0.01
  )
  stopifnot(abs(modeNative[[1]] - modeR[[1]]) < 1e-9)
  stopifnot(max(abs(modeNative[[2]] - modeR[[2]])) < 1e-12)
  stopifnot(max(abs(modeNative[[3]] - modeR[[3]])) < 1e-12)
  stopifnot(max(abs(modeNative[[4]] - modeR[[4]])) < 1e-9)
  stopifnot(max(abs(modeNative[[5]] - modeR[[5]])) < 1e-8)
  stopifnot(identical(modeNativeSeed, .Random.seed))
  rye.nativeOptimizer = TRUE
}

## L'Ecuyer-CMRG state and draws must also round-trip exactly through Rust.
RNGkind('L\'Ecuyer-CMRG', normal.kind = 'Inversion', sample.kind = 'Rejection')
set.seed(456)
lecuyerNative = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 200, sd = 0.01
)
lecuyerNativeSeed = .Random.seed
rye.nativeOptimizer = FALSE
set.seed(456)
lecuyerR = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 200, sd = 0.01
)
lecuyerRSeed = .Random.seed
stopifnot(abs(lecuyerNative[[1]] - lecuyerR[[1]]) < 1e-9)
stopifnot(max(abs(lecuyerNative[[2]] - lecuyerR[[2]])) < 1e-12)
stopifnot(max(abs(lecuyerNative[[3]] - lecuyerR[[3]])) < 1e-12)
stopifnot(max(abs(lecuyerNative[[4]] - lecuyerR[[4]])) < 1e-9)
stopifnot(max(abs(lecuyerNative[[5]] - lecuyerR[[5]])) < 1e-8)
stopifnot(identical(lecuyerNativeSeed, lecuyerRSeed))
rye.nativeOptimizer = TRUE

## Unsupported R generators must fall back without consuming RNG state during
## the failed native attempt.
RNGkind('Wichmann-Hill', normal.kind = 'Inversion', sample.kind = 'Rejection')
set.seed(789)
fallbackResult = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 100, sd = 0.01
)
fallbackSeed = .Random.seed
rye.nativeOptimizer = FALSE
set.seed(789)
directResult = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 100, sd = 0.01
)
directSeed = .Random.seed
stopifnot(identical(fallbackResult, directResult))
stopifnot(identical(fallbackSeed, directSeed))
rye.nativeOptimizer = TRUE

## Deterministic math must reproduce both outputs and final RNG state exactly.
RNGkind('Mersenne-Twister', normal.kind = 'Inversion', sample.kind = 'Rejection')
rye.nativeDeterministicMath = TRUE
set.seed(321)
deterministicFirst = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 200, sd = 0.01
)
deterministicFirstSeed = .Random.seed
set.seed(321)
deterministicSecond = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 200, sd = 0.01
)
stopifnot(identical(deterministicFirst, deterministicSecond))
stopifnot(identical(deterministicFirstSeed, .Random.seed))
rye.nativeDeterministicMath = FALSE

## Native execution must initialize and return R state when the session has not
## generated a random number yet.
rm('.Random.seed', envir = .GlobalEnv)
unseededResult = rye.gibbs(
  referenceX, referenceFam, referenceGroups,
  alpha, TRUE, weight, TRUE, iterations = 1, sd = 0.01
)
stopifnot(length(unseededResult) == 5)
stopifnot(is.integer(.Random.seed), length(.Random.seed) == 626)

## Every logical attempt receives the same deterministic L'Ecuyer stream,
## regardless of whether attempts run serially or in forked workers.
if (.Platform$OS.type != 'windows') {
  RNGkind('Mersenne-Twister', normal.kind = 'Inversion', sample.kind = 'Rejection')
  set.seed(654)
  callerSeed = .Random.seed
  serialResult = rye.optimize(
    referenceX, referenceFam,
    referencePops = groups, referenceGroups = referenceGroups,
    alpha = alpha, weight = weight,
    attempts = 4, iterations = 50, rounds = 3, threads = 1,
    seed = 2026
  )
  stopifnot(identical(callerSeed, .Random.seed))
  parallelResult = rye.optimize(
    referenceX, referenceFam,
    referencePops = groups, referenceGroups = referenceGroups,
    alpha = alpha, weight = weight,
    attempts = 4, iterations = 50, rounds = 3, threads = 2,
    seed = 2026
  )
  stopifnot(identical(serialResult, parallelResult))
  stopifnot(identical(callerSeed, .Random.seed))

  set.seed(777)
  implicitSerial = rye.optimize(
    referenceX, referenceFam,
    referencePops = groups, referenceGroups = referenceGroups,
    alpha = alpha, weight = weight,
    attempts = 4, iterations = 50, rounds = 3, threads = 1
  )
  implicitSerialSeed = .Random.seed
  set.seed(777)
  implicitParallel = rye.optimize(
    referenceX, referenceFam,
    referencePops = groups, referenceGroups = referenceGroups,
    alpha = alpha, weight = weight,
    attempts = 4, iterations = 50, rounds = 3, threads = 2
  )
  stopifnot(identical(implicitSerial, implicitParallel))
  stopifnot(identical(implicitSerialSeed, .Random.seed))
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
