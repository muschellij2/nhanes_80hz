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

# types = c("agcounts", "GGIR")
types = "GGIR"
type = types[1]

for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  file = idf$csv_file
  acc_csv_file = idf$acc_csv_file
  for (type in types) {
    type = match.arg(type, types)
    cal_file = switch(
      type,
      agcounts = idf$calibrated_file,
      GGIR = idf$ggir_calibrated_file)
    params_file = switch(
      type,
      agcounts = idf$calibration_params_file,
      GGIR = idf$ggir_calibration_params_file)
    outfiles = c(
      cal_file,
      params_file)
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

      if (type == "agcounts") {
        message("Creating Matrix")
        mat = as.matrix(data[, xyz])
        message("Running gcalibrate")
        gc()
        # calibrated = agcalibrate(df, verbose = TRUE)
        C <- try({
          agcounts:::gcalibrateC(dataset = mat, sf = 80L)
        })
        rm(mat)
      } else if (type == "GGIR") {
        message("Inspecting file")
        ggir_I <- GGIR::g.inspectfile(datafile = idf$acc_csv_file)
        message("Running g.calibrate")
        C <- try({
          GGIR::g.calibrate(
            datafile = idf$acc_csv_file,
            use.temp = FALSE,
            printsummary = FALSE,
            inspectfileobject = ggir_I)
        })
      }
      if (inherits(C, "try-error")) {
        rm(data)
        gc()
        next
      }
      # message("Running GC after gcalibrate")
      # gc()
      message("Creating Calibration Table")
      cmat = tibble(
        scale = C$scale,
        offset = C$offset,
        axis = xyz
      )
      cmat$cal_error_start = C$cal.error.start
      cmat$cal_error_end = C$cal.error.end
      cmat$nhoursused = C$nhoursused
      cmat$npoints = C$npoints
      names(C$offset) = xyz
      names(C$scale) = xyz
      write_csv_gz(cmat, params_file)

      message("Calibrating Data")
      for (icol in xyz) {
        data[,icol] <- round(
          (data[,icol] - (-C$offset[icol])) / (1/C$scale[icol]),
          4)
      }
      write_csv_gz(data, cal_file)
      rm(data)
      rm(C)
    }
  }
}
