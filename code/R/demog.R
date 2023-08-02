library(haven)
library(dplyr)
library(nhanesA)
table = "DEMO_G"

read_and_relabel = function(table) {
  file = here::here("data", "raw", paste0(table, ".XPT"))
  df = haven::read_xpt(file)
  nh_table = table
  if (table == "DEMO_Y") {
    nh_table = "Y_DEMO"
  }
  df = nhanesA::nhanesTranslate(nh_table = nh_table, data = df,
                                colnames = colnames(df))
  cn = colnames(df)
  for (icn in cn) {
    attr(df[[icn]], "variable_name") = icn
  }
  cn = sapply(df, attr, "label")
  cn = janitor::make_clean_names(cn)
  colnames(df) = cn

  df = df %>%
    dplyr::mutate(
      nh_table = table,
      wave = sub("demo_", "", tolower(nh_table)),
      version = paste0("pax_", wave)
    )
  df
}
waves = c("DEMO_G", "DEMO_H", "DEMO_Y")
dfs = lapply(waves, read_and_relabel)
names(dfs) = waves
