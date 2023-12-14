library(magrittr)
library(dplyr)
library(readr)
library(SummarizedActigraphy)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

df = df %>%
  filter(!file.exists(steps_1s_file) & file.exists(acc_steps_1s_file))

# id is bad: 71917
i = 1
for (i in seq_len(nrow(df))) {
  idf = df[i,]
  if (!file.exists(idf$steps_1s_file)) {
    check_cols = c("time", "X", "Y", "Z")
    data = SummarizedActigraphy::read_acc_csv(file = idf$acc_steps_1s_file,
                                              progress = FALSE)
    data = data$data
    colnames(data) = c("time", "Y", "X", "Z", "steps")
    data = data[, c(check_cols, "steps")]
    check_data = readr::read_csv(
      idf$counts_1s_file,
      col_types = cols(
        HEADER_TIMESTAMP = col_datetime(format = ""),
        X = col_double(),
        Y = col_double(),
        Z = col_double(),
        AC = col_double()
      ),
      progress = FALSE,
      show_col_types = FALSE,
      num_threads = 1L)
    readr::stop_for_problems(check_data)
    check_data = check_data %>%
      dplyr::rename(time = any_of(c("time", "HEADER_TIMESTAMP", "HEADER_TIME_STAMP")))
    check_time = all(unique(diff(check_data$time)) == 1) &&
      all(unique(diff(data$time)) == 1)
    check_dim = nrow(check_data) == nrow(data)
    the_check = all.equal(check_data[, check_cols], data[, check_cols])
    is_identical = isTRUE(the_check)
    if (!check_time || ! check_dim) {
      if (!check_time) {
        message(paste0(idf$id, ": Times aren't just 1s"))
      }
      if (!check_dim) {
        message(paste0(idf$id, ": Dims aren't right. nrow(actilife): ",
                       nrow(data), ", nrow(python):", nrow(check_data)))
      }
      next
    }
    if (!is_identical) {
      print(the_check)
      checker = rowSums(check_data[, check_cols] != data[, check_cols],
                        na.rm = TRUE) > 0
      print("From Python")
      print(check_data[checker, ])
      print("From ActiLife")
      print(data[checker, ])

      message(paste0(idf$id, ": data isn't same - not writing"))
    }
    # else {
      write_csv_gz(df = data, file = idf$steps_1s_file,
                   progress = FALSE)
    # }
  }
}
