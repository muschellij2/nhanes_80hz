library(magrittr)
library(dplyr)
library(agcounts)
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

xdf = df

# df = df %>%
#   dplyr::filter(file.exists(csv_file))

max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(paste0("File ", index, " of ", max_n, ": ", idf$csv_file))
  files = list(
    counts_60s_file = idf$counts_60s_from_30Hz_file
  )
  dir.create(dirname(idf$counts_60s_from_30Hz_file), showWarnings = FALSE, recursive = TRUE)

  if (!all(file.exists(unlist(files)))) {
    if (!file.exists(idf$counts_60s_from_30Hz_file)) {
      df30 = readr::read_csv(idf$csv30_file, progress = FALSE)
      df30 = df30 %>%
        rename(time = HEADER_TIMESTAMP)

      attr(df30, "sample_rate") = 30L
      agresult = agcounts::calculate_counts(
        df30,
        epoch = 60L,
        lfe_select = FALSE)
      agresult = agresult %>%
        rename(HEADER_TIMESTAMP = time,
               Y = Axis1,
               X = Axis2,
               Z = Axis3,
               AC = Vector.Magnitude) %>%
        select(HEADER_TIMESTAMP, X, Y, Z, AC)
      readr::write_csv(
        agresult,
        gzfile(idf$counts_60s_from_30Hz_file, compress = 9L)
      )
    }
    # df = read_80hz(idf$csv_file)
    # x = agcounter::get_counts(
    #   df,
    #   sample_rate = 80L,
    #   epoch_in_seconds = 1L
    # )
    # doing this so .Last.value isn't maintained
    rm(x)
  }


  files = list(
    counts_60s_lfe_file = idf$counts_60s_lfe_from_30Hz_file
  )
  dir.create(dirname(idf$counts_60s_lfe_from_30Hz_file), showWarnings = FALSE, recursive = TRUE)

  if (!all(file.exists(unlist(files)))) {
    if (!file.exists(idf$counts_60s_lfe_from_30Hz_file)) {
      df30 = readr::read_csv(idf$csv30_file, progress = FALSE)
      df30 = df30 %>%
        rename(time = HEADER_TIMESTAMP)

      attr(df30, "sample_rate") = 30L
      try({
        agresult = agcounts::calculate_counts(
          df30,
          epoch = 60L,
          lfe_select = TRUE)
        agresult = agresult %>%
          rename(HEADER_TIMESTAMP = time,
                 Y = Axis1,
                 X = Axis2,
                 Z = Axis3,
                 AC = Vector.Magnitude) %>%
          select(HEADER_TIMESTAMP, X, Y, Z, AC)
        readr::write_csv(
          agresult,
          gzfile(idf$counts_60s_lfe_from_30Hz_file, compress = 9L)
        )
      })
    }
    # df = read_80hz(idf$csv_file)
    # x = agcounter::get_counts(
    #   df,
    #   sample_rate = 80L,
    #   epoch_in_seconds = 1L
    # )
    # doing this so .Last.value isn't maintained
    rm(x)
  }

}
