library(dplyr)
library(here)
source("code/R/utils.R")
source("code/R/helper_functions.R")

nh_table = "PAXDAY_G"

write_individual_data = function(df = NULL, nh_table,
                                 verbose = TRUE, ...) {
  if (is.null(df)) {
    df = read_daily_min(nh_table)
  }

  nh_table = nh_table_name(nh_table)
  outdir = table_to_outdir(nh_table)
  stopifnot(!is.na(outdir))
  wave = get_wave(nh_table)
  pax_name = paste0("pax_", wave)
  data_dir = here::here("data", outdir)

  table_dir = file.path(data_dir, pax_name)
  dir.create(table_dir, showWarnings = FALSE, recursive = TRUE)

  uids = unique(df$SEQN)
  iid = uids[1]
  for (iid in uids) {
    outfile = file.path(table_dir, paste0(iid, ".csv.gz"))
    if (verbose) {
      print(outfile)
    }
    if (!file.exists(outfile)) {
      idf = df %>%
        dplyr::filter(SEQN == iid)
      write_csv_gz(idf, outfile)
    }
  }
}

write_individual_data(nh_table = "PAXDAY_G")
write_individual_data(nh_table = "PAXDAY_H")
write_individual_data(nh_table = "PAXDAY_Y")
