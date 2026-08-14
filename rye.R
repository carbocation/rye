#!/usr/bin/env Rscript

ryeLauncherFile = tryCatch(sys.frame(1)$ofile, error = function(error) NULL)
if (is.null(ryeLauncherFile)) {
  scriptArgument = grep('^--file=', commandArgs(trailingOnly = FALSE), value = TRUE)
  ryeLauncherFile = if (length(scriptArgument)) {
    sub('^--file=', '', scriptArgument[[1]])
  } else {
    file.path(getwd(), 'rye.R')
  }
}
ryeRepository = dirname(normalizePath(ryeLauncherFile))
sys.source(file.path(ryeRepository, 'R', 'rye.R'), envir = .GlobalEnv)
rye.loadDevelopmentNative(ryeRepository)

if (sys.nframe() == 0) {
  rye.main()
}
