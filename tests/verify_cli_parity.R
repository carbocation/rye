#!/usr/bin/env Rscript

arguments <- commandArgs(trailingOnly = TRUE)
repository <- normalizePath(if (length(arguments)) arguments[[1]] else ".")
binary <- if (length(arguments) >= 2) {
  normalizePath(arguments[[2]])
} else {
  normalizePath(file.path(repository, "target/release/rye"))
}
source(file.path(repository, "rye.R"))
stopifnot(rye.loadDevelopmentNative(repository))

directory <- tempfile("rye-cli-parity-")
dir.create(directory)
on.exit(unlink(directory, recursive = TRUE), add = TRUE)
rPrefix <- file.path(directory, "r")
serialPrefix <- file.path(directory, "rust-serial")
parallelPrefix <- file.path(directory, "rust-parallel")

invisible(capture.output(rye(
  eigenvec_file = file.path(repository, "examples/example.eigenvec"),
  eigenval_file = file.path(repository, "examples/example.eigenval"),
  pop2group_file = file.path(repository, "examples/pop2group.txt"),
  output_file = rPrefix,
  threads = 1, pcs = 20, optim_rounds = 3, optim_iter = 20,
  attempts = 4, seed = 2026
)))

runBinary <- function(prefix, threads) {
  output <- system2(binary, c(
    paste0("--eigenvec=", file.path(repository, "examples/example.eigenvec")),
    paste0("--eigenval=", file.path(repository, "examples/example.eigenval")),
    paste0("--pop2group=", file.path(repository, "examples/pop2group.txt")),
    paste0("--output=", prefix),
    paste0("--threads=", threads),
    "--rounds=3", "--iter=20", "--attempts=4", "--seed=2026"
  ), stdout = TRUE, stderr = TRUE)
  stopifnot(is.null(attr(output, "status")))
}
runBinary(serialPrefix, 1)
runBinary(parallelPrefix, 4)

qSuffix <- "-20.7.Q"
famSuffix <- "-20.fam"
readBytes <- function(path) readBin(path, "raw", file.info(path)$size)
stopifnot(identical(
  readBytes(paste0(serialPrefix, qSuffix)),
  readBytes(paste0(parallelPrefix, qSuffix))
))
stopifnot(identical(
  readBytes(paste0(serialPrefix, famSuffix)),
  readBytes(paste0(parallelPrefix, famSuffix))
))
stopifnot(identical(
  readBytes(paste0(rPrefix, famSuffix)),
  readBytes(paste0(serialPrefix, famSuffix))
))

rQ <- read.table(paste0(rPrefix, qSuffix), header = TRUE, row.names = 1, check.names = FALSE)
rustQ <- read.table(
  paste0(serialPrefix, qSuffix), header = TRUE, row.names = 1, check.names = FALSE
)
stopifnot(identical(dimnames(rQ), dimnames(rustQ)))
maximumDifference <- max(abs(as.matrix(rQ) - as.matrix(rustQ)))
stopifnot(maximumDifference <= 2e-15)
message("Standalone CLI parity passed; maximum Q difference: ", maximumDifference)
