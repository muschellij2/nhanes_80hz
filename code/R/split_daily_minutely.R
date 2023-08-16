library(dplyr)
library(here)
source(here::here("code/R/utils.R"))
source(here::here("code/R/helper_functions.R"))

nh_table = "PAXMIN_H"
df = NULL

write_full_csv = function( nh_table, df = NULL, ...) {
  stopifnot(length(nh_table) == 1)

  file = daily_min_file(nh_table)
  file = sub("[.]XPT$", ".csv.gz", file)

  if (!file.exists(file)) {
    if (is.null(df)) {
      df = read_daily_min(nh_table)
    }
    write_csv_gz(df, file)
  }
}

write_individual_data = function(df = NULL, nh_table,
                                 verbose = TRUE, ...) {
  stopifnot(length(nh_table) == 1)

  nh_table = nh_table_name(nh_table)
  outdir = table_to_outdir(nh_table)
  stopifnot(!is.na(outdir))
  wave = get_wave(nh_table)
  pax_name = paste0("pax_", wave)
  data_dir = here::here("data", outdir)

  table_dir = file.path(data_dir, pax_name)
  dir.create(table_dir, showWarnings = FALSE, recursive = TRUE)


  uids = NULL
  if (is.null(df)) {
    day_table = sub("PAXMIN", "PAXDAY", nh_table)
    day_df = read_daily_min(day_table)
    uids = unique(day_df$SEQN)
    rm(day_df)
    outfiles = file.path(table_dir, paste0(uids, ".csv.gz"))
    fe = file.exists(outfiles)
    if (all(fe)) {
      return(NULL)
    }
  }

  if (is.null(df)) {
    df = read_daily_min(nh_table)
  }
  if (is.null(uids)) {
    uids = unique(df$SEQN)
  }

  outfiles = file.path(table_dir, paste0(uids, ".csv.gz"))
  fe = file.exists(outfiles)
  uids = uids[!fe]
  iid = uids[1]
  df = split(df, df$SEQN)
  for (iid in uids) {
    outfile = file.path(table_dir, paste0(iid, ".csv.gz"))
    if (verbose) {
      print(outfile)
    }
    if (!file.exists(outfile)) {
      idf = df[[as.character(iid)]]
      write_csv_gz(idf, outfile)
    }
  }
}

write_individual_data(nh_table = "PAXDAY_G")
write_individual_data(nh_table = "PAXDAY_H")
write_individual_data(nh_table = "PAXDAY_Y")

write_individual_data(nh_table = "PAXMIN_G")
write_individual_data(nh_table = "PAXMIN_H")
write_individual_data(nh_table = "PAXMIN_Y")


write_full_csv(nh_table = "PAXDAY_G")
write_full_csv(nh_table = "PAXDAY_H")
write_full_csv(nh_table = "PAXDAY_Y")

write_full_csv(nh_table = "PAXMIN_G")
write_full_csv(nh_table = "PAXMIN_H")
write_full_csv(nh_table = "PAXMIN_Y")
