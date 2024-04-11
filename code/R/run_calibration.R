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
  outfile = idf$calibrated_file
  dir.create(dirname(outfile), showWarnings = FALSE, recursive = TRUE)
  print(file)

  if (!all(file.exists(outfile))) {
    data = read_80hz(file, progress = FALSE)
    data = data %>%
      dplyr::rename(time = HEADER_TIMESTAMP)
    # needed for fix of agcounts
    # PR at https://github.com/bhelsel/agcounts/pull/32
    data = as.data.frame(data)
    attr(data, "sample_rate") = 80L
    attr(data, "last_sample_time") = max(df$time)

    # calibrated = agcalibrate(df, verbose = TRUE)
    C <- agcounts:::gcalibrateC(dataset = as.matrix(data[, c("X", "Y", "Z")]), sf = 80L)

    xyz = c("X", "Y", "Z")
    data[, xyz] <- scale(data[, xyz], center = -C$offset, scale = 1/C$scale)

    # I <- GGIR::g.inspectfile(datafile = file)
    # C <- GGIR::g.calibrate(datafile = file,
    #                        use.temp = FALSE,
    #                        printsummary = FALSE,
    #                        inspectfileobject = I)

    write_csv_gz(data, outfile)
    rm(data)
  }
}
