test_that("the packaged Rust accelerator loads", {
  expect_true(get("rye.nativeSolver", envir = asNamespace("rye")))
  expect_true(get("rye.nativeOptimizer", envir = asNamespace("rye")))
  expect_true(
    get("rye.nativeKernel", envir = asNamespace("rye")) %in%
      c("scalar", "NEON", "AVX2", "AVX-512")
  )
})

test_that("packaged prediction returns normalized nonnegative coefficients", {
  x = matrix(
    c(0.8, 0.2, 0.7, 0.3, 0.1, 0.9),
    nrow = 3,
    dimnames = list(c("a", "b", "c"), c("pc1", "pc2"))
  )
  means = matrix(
    c(0.9, 0.1, 0.1, 0.9),
    nrow = 2,
    dimnames = list(c("left", "right"), c("pc1", "pc2"))
  )
  result = rye.predict(x, means, weight = c(1, 1))
  expect_equal(dim(result), c(3L, 2L))
  expect_equal(unname(rowSums(result)), rep(1, 3), tolerance = 1e-12)
  expect_true(all(result >= 0))
})

test_that("Rust worker streams match R nextRNGStream", {
  streams = get("rye.lecuyerStreams", envir = asNamespace("rye"))(42L, 4L)

  oldKind = RNGkind()
  oldSeedExists = exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  oldSeed = if (oldSeedExists) .Random.seed else NULL
  on.exit({
    do.call(RNGkind, as.list(oldKind))
    if (oldSeedExists) {
      assign(".Random.seed", oldSeed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  })
  RNGkind("L'Ecuyer-CMRG", normal.kind = "Inversion")
  set.seed(42)
  expected = matrix(0L, nrow = 7L, ncol = 4L)
  state = .Random.seed
  for (index in seq_len(ncol(expected))) {
    expected[ , index] = state
    if (index < ncol(expected)) state = parallel::nextRNGStream(state)
  }
  expect_identical(streams, expected)
})

optimizerFixture = function() {
  samples = c("s1", "s2", "s3", "s4")
  groups = c("left", "right")
  X = matrix(
    c(0.9, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.9),
    nrow = 4,
    dimnames = list(samples, c("pc1", "pc2"))
  )
  fam = cbind(
    population = c("left", "left", "right", "right"),
    id = samples
  )
  rownames(fam) = samples
  list(
    X = X,
    fam = fam,
    groups = groups,
    referenceGroups = setNames(groups, groups),
    alpha = setNames(rep(0.001, length(groups)), groups),
    weight = c(1, 0.5)
  )
}

runOptimizerFixture = function(fixture, threads, seed = NULL) {
  capture.output(result <- rye.optimize(
    fixture$X,
    fixture$fam,
    referencePops = fixture$groups,
    referenceGroups = fixture$referenceGroups,
    alpha = fixture$alpha,
    weight = fixture$weight,
    attempts = 4,
    iterations = 25,
    rounds = 3,
    threads = threads,
    seed = seed
  ))
  result
}

test_that("an explicit optimizer seed is reproducible and preserves R state", {
  fixture = optimizerFixture()
  set.seed(91)
  state = .Random.seed
  first = runOptimizerFixture(fixture, threads = 1, seed = 2026)
  expect_identical(.Random.seed, state)
  second = runOptimizerFixture(fixture, threads = 1, seed = 2026)
  expect_identical(first, second)
  expect_identical(.Random.seed, state)
})

test_that("optimizer streams are independent of worker count", {
  skip_on_os("windows")
  fixture = optimizerFixture()

  serial = runOptimizerFixture(fixture, threads = 1, seed = 2026)
  twoWorkers = runOptimizerFixture(fixture, threads = 2, seed = 2026)
  fourWorkers = runOptimizerFixture(fixture, threads = 4, seed = 2026)
  expect_identical(serial, twoWorkers)
  expect_identical(twoWorkers, fourWorkers)

  set.seed(2026)
  implicitSerial = runOptimizerFixture(fixture, threads = 1)
  set.seed(2026)
  implicitParallel = runOptimizerFixture(fixture, threads = 2)
  expect_identical(implicitSerial, implicitParallel)
})
