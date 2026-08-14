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
