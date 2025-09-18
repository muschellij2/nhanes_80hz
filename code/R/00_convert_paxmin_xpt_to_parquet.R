# install.packages(c("haven", "arrow", "duckdb", "DBI", "dplyr"))
library(haven)
library(arrow)
library(dplyr)
library(here)

df = tibble(
  id = c("PAXMIN_Y", "PAXMIN_G", "PAXMIN_H"),
  file = here::here("data", "raw_min", paste0(id, ".XPT")),
  outfile = here::here("data", "raw_min", paste0(id, ".parquet"))
)

iid = 1
for (iid in seq(nrow(df))) {
  idf = df[iid,]
  print(idf)

  if (!file.exists(idf$outfile)) {
    # 1) Read once (this loads into memory)
    x <- read_xpt(idf$file)  # consider haven::zap_labels() / as_factor() as needed

    # (Optional) Preserve SAS labels as factors or save a codebook before writing Parquet
    # xx <- haven::as_factor(x, only_labelled = TRUE)

    # 2) Write Parquet with good compression & row groups (tune row_group_size for your size)
    write_parquet(
      x,
      idf$outfile
    )
  }
  rm(list = "x")
}
