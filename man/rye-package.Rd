\name{rye-package}
\alias{rye-package}
\docType{package}
\title{Rapid Ancestry Estimation}
\description{
Rye estimates global genetic ancestry from principal-component data using a
Rust-accelerated non-negative least-squares optimizer.
}
\details{
The package builds its Rust numerical core during installation. The command
line workflow is available through \code{rye.main()}, while the matrix-level
API is exposed by \code{rye.predict()} and \code{rye.optimize()}.
}
