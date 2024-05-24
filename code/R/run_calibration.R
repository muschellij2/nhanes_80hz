library(janitor)
library(tibble)
library(GGIR)
library(agcounts)
library(dplyr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
zero_df = readr::read_csv(here::here("data/raw/all_zero.csv.gz"))
df = left_join(df,
               zero_df %>%
                 select(id, version, all_zero) %>%
                 mutate(id = as.character(id)))

xdf = df

ifold = get_fold()

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

i = 1

for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  file = idf$csv_file
  acc_csv_file = idf$acc_csv_file
  outfiles = c(idf$calibrated_file,
               idf$ggir_calibrated_file,
               idf$calibration_params_file,
               idf$ggir_calibration_params_file)
  sapply(outfiles, function(outfile) {
    dir.create(dirname(outfile), showWarnings = FALSE, recursive = TRUE)
  })
  print(file)

  if (!all(file.exists(outfiles))) {
    data = read_80hz(file, progress = FALSE)
    data = data %>%
      dplyr::rename(time = HEADER_TIMESTAMP)
    # needed for fix of agcounts
    # PR at https://github.com/bhelsel/agcounts/pull/32
    data = as.data.frame(data)
    attr(data, "sample_rate") = 80L
    attr(data, "last_sample_time") = max(data$time)
    xyz = c("X", "Y", "Z")

    mat = as.matrix(data[, xyz])
    # calibrated = agcalibrate(df, verbose = TRUE)
    C <- try({
      agcounts:::gcalibrateC(dataset = mat, sf = 80L)
      })
    if (inherits(C, "try-error")) {
      rm(data)
      rm(mat)
      gc()
      next
    }
    gc()
    cmat = tibble(
      scale = C$scale,
      offset = C$offset,
      axis = xyz
    )
    names(C$offset) = xyz
    names(C$scale) = xyz
    write_csv_gz(cmat, idf$calibration_params_file)

    for (icol in xyz) {
      data[,icol] <- round(
        (mat[,icol] - (-C$offset[icol])) / (1/C$scale[icol]),
        4)
    }
    # data[,xyz] <- round(
    #   scale(mat[,xyz], center = -C$offset, scale = 1/C$scale),
    #   4)
    write_csv_gz(data, idf$calibrated_file)
    rm(mat)
    gc()

    ggir_I <- GGIR::g.inspectfile(datafile = idf$acc_csv_file)
    ggir_C <- try({
      GGIR::g.calibrate(datafile = idf$acc_csv_file,
                        use.temp = FALSE,
                        printsummary = FALSE,
                        inspectfileobject = ggir_I)
    })
    if (inherits(ggir_C, "try-error")) {
      rm(data)
      rm(mat)
      gc()
      next
    }
    cmat = tibble(
      scale = ggir_C$scale,
      offset = ggir_C$offset,
      axis = xyz
    )
    names(ggir_C$offset) = xyz
    names(ggir_C$scale) = xyz
    write_csv_gz(cmat, idf$ggir_calibration_params_file)


    ## Using GGIR Derived
    # data[,xyz] <- round(
    #   scale(mat[,xyz], center = -ggir_C$offset, scale = 1/ggir_C$scale),
    #   4)
    for (icol in xyz) {
      data[,icol] <- round(
        (data[,icol] - (-ggir_C$offset[icol])) / (1/ggir_C$scale[icol]),
        4)
    }
    write_csv_gz(data, idf$ggir_calibrated_file)
    rm(data)
  }
}
