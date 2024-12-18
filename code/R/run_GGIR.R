#!/usr/bin/env Rscript
library(dplyr)
library(GGIR)
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

# ggir_files = df %>%
#   select(id, starts_with("ggir_part"))
# x = ggir_files %>%
#   select(-id) %>%
#   as.matrix()
# res = array(file.exists(x), dim = dim(x), dimnames = dimnames(x))
# file.remove(x[which(rowSums(!res) > 0),])

# 64750
max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(paste0(index, " of ", max_n))
  print(idf$csv_file)

  dir_output = here::here("data", "GGIR", idf$version)
  sapply(dir_output, dir.create, showWarnings = FALSE, recursive = TRUE)



  res = try({
    GGIR::GGIR(
      datadir = idf$csv_file,
      outputdir = dir_output,
      studyname = idf$id,
      print.filename = TRUE,
      desiredtz = "UTC",
      configtz = "UTC",
      sensor.location = "wrist",
      verbose = TRUE,
      minimumFileSizeMB = 0, # don't let small gz files be skipped
      do.parallel = FALSE,
      dynrange = 6L, # dynamic range of 6g (https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/PAX80_G.htm)
      idloc = 6, # file name gives the ID before the "."

      #=====================
      # read.myacc.csv arguments for reading in CSVS
      #=====================
      rmc.nrow = Inf,
      rmc.skip = 0,
      rmc.dec = ".",
      rmc.firstrow.acc = 2,
      rmc.firstrow.header = NULL,
      rmc.header.length = NULL,
      rmc.col.acc = 2:4,
      rmc.col.temp = NULL,
      rmc.col.time = 1,
      rmc.unit.acc = "g",
      rmc.unit.time = "character",
      rmc.format.time = "%Y-%m-%dT%H:%M:%OS",
      rmc.bitrate = NULL,
      rmc.dynamic_range = 6,
      rmc.unsignedbit = TRUE,
      rmc.headername.sf = NULL,
      rmc.headername.sn = NULL,
      rmc.headername.recordingid = NULL,
      rmc.header.structure = NULL,
      rmc.check4timegaps = FALSE,
      rmc.col.wear = NULL,
      rmc.doresample = FALSE,
      rmc.scalefactor.acc = 1,
      rmc.sf = 80L
    )
  })

}


