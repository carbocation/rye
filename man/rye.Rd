\name{rye}
\alias{rye}
\alias{rye.absoluteError}
\alias{rye.basePopulationMeans}
\alias{rye.gibbs}
\alias{rye.main}
\alias{rye.optimize}
\alias{rye.parameterizePopulationMeans}
\alias{rye.populationMeans}
\alias{rye.predict}
\alias{rye.scale}
\title{Estimate Global Genetic Ancestry}
\description{
Read PCA inputs, optimize reference-population parameters, and estimate
nonnegative ancestry coefficients using Rye's Rust numerical core.
}
\usage{
rye(eigenvec_file = NULL, eigenval_file = NULL, pop2group_file = NULL,
    output_file = NULL, threads = 4, pcs = 20, optim_rounds = 200,
    optim_iter = 100, attempts = 4)

rye.scale(X = NULL)

rye.basePopulationMeans(X = NULL, fam = NULL, fn = median,
    referenceGroups = NULL)

rye.parameterizePopulationMeans(means = NULL, alpha = NULL, weight = NULL)

rye.populationMeans(X = NULL, fam = NULL, alpha = NULL, weight = NULL,
    fn = median, referenceGroups = NULL)

rye.predict(X = NULL, means = NULL, weight = NULL,
    referenceGroups = NULL, warmStart = NULL)

rye.absoluteError(expected = NULL, predicted = NULL)

rye.gibbs(X = NULL, fam = NULL, referenceGroups = NULL, alpha = NULL,
    optimizeAlpha = TRUE, weight = NULL, optimizeWeight = TRUE,
    iterations = 100, sd = 0.0001, baseMeans = NULL)

rye.optimize(X = NULL, fam = NULL, referencePops = NULL,
    referenceGroups = NULL, alpha = NULL, optimizeAlpha = TRUE,
    weight = NULL, optimizeWeight = TRUE, attempts = 4,
    iterations = 100, rounds = 25, threads = 1, startSD = 0.005,
    endSD = 0.001, populationError = FALSE)

rye.main(args = commandArgs(trailingOnly = TRUE))
}
\arguments{
  \item{eigenvec_file, eigenval_file, pop2group_file}{Input PCA and grouping files.}
  \item{output_file}{Prefix for generated ancestry and family files.}
  \item{threads}{Number of independent attempts that may run concurrently.}
  \item{pcs}{Number of principal components to use.}
  \item{optim_rounds, optim_iter, attempts, iterations, rounds}{Optimization controls.}
  \item{X}{Numeric sample-by-feature matrix.}
  \item{fam}{Sample population and identifier matrix.}
  \item{fn}{Summary function used for reference populations.}
  \item{referenceGroups, referencePops}{Population-to-group mappings.}
  \item{means}{Reference-population feature matrix.}
  \item{alpha, weight}{Shrinkage and feature-weight parameters.}
  \item{warmStart}{Optional coefficient matrix from a previous solve.}
  \item{expected, predicted}{Values used to calculate absolute error.}
  \item{optimizeAlpha, optimizeWeight}{Whether each parameter family is optimized.}
  \item{sd, startSD, endSD}{Proposal standard-deviation controls.}
  \item{baseMeans}{Optional precomputed reference-population medians.}
  \item{populationError}{Compatibility argument retained by \code{rye.optimize}.}
  \item{args}{Command-line arguments accepted by Rye.}
}
\value{
Depending on the function, a coefficient matrix, optimized parameter list, or
generated output files. \code{rye.main()} is called for its side effects.
}
