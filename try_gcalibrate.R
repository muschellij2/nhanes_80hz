options(repos = "https://cloud.r-project.org/")
library(GGIR)
library(agcounts)
library(dplyr)
library(readr)
options(digits.secs = 3)

file = "~/Dropbox/Projects/nhanes_80hz/data/csv/pax_h/73587.csv.gz"
if (!file.exists(file)) {
  url = "https://figshare.com/ndownloader/files/46780216"
  file = tempfile(fileext = ".csv.gz")
  download.file(url, destfile = file, mode = "wb")
}

data = readr::read_csv(
  file,
  col_types = cols(
    HEADER_TIMESTAMP = col_datetime(format = ""),
    X = col_double(),
    Y = col_double(),
    Z = col_double()
  )
)
probs = readr::problems(data)
stopifnot(nrow(probs) == 0)
readr::stop_for_problems(data)

data = data %>%
  dplyr::rename(time = HEADER_TIMESTAMP)
# needed for fix of agcounts
# PR at https://github.com/bhelsel/agcounts/pull/32
data = as.data.frame(data)
attr(data, "sample_rate") = 80L
attr(data, "last_sample_time") = max(data$time)
xyz = c("X", "Y", "Z")
gc()

message("Creating Matrix")
mat = as.matrix(data[, xyz])
message("Running gcalibrate")
# calibrated = agcalibrate(df, verbose = TRUE)
for (i in 1:3) {
  print(i)
  C <- try({
    agcounts:::gcalibrateC(dataset = mat, sf = 80L)
  })
  gc()
}
