library(googledrive)
library(googlesheets4)
library(janitor)
library(dplyr)
id = "13GJ85-itB2VfKhYXXl6ZZCmzAWfb5skw3rqfxD8U9VE"

normalize_table_name = function(nh_table) {
  table = toupper(nh_table)

  y_table = grepl("^Y_", table)
  if (any(y_table)) {
    table[y_table] = sub("^Y_(.*)", "\\1_Y", table[y_table])
  }
  table
}


x = read_sheet(ss = id)

df = x %>% clean_names()
df = df %>%
  select(table, variable)
df = df %>%
  mutate(table = normalize_table_name(table),
         variable = toupper(variable))
files = paste0("data/demographics/", df$table, ".XPT")
stopifnot(all(file.exists(files)))

ss = split(df, df$table)
ss = lapply(ss, function(r) {
  table = unique(r$table)
  file = paste0("data/demographics/", table, ".XPT")
  if (!"SEQN" %in% r$variable) {
    r = bind_rows(r, data.frame(table = unique(table), variable = "SEQN"))
  }
  r$file = file
  r
})
df = dplyr::bind_rows(ss)
df = df %>%
  tidyr::nest(data = any_of("variable"))

readr::write_rds(df, here::here("data/demographics/data_column_subsets.rds"))
