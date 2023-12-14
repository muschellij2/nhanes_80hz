library(haven)
library(dplyr)
library(nhanesA)
library(rlang)
source("code/R/utils.R")
table = "DEMO_G"


attach_translations = function(df, translations) {
  cn = colnames(df)
  for (icn in cn) {
    attr(df[[icn]], "variable_name") = icn
    attr(df[[icn]], "translation") = translations[[icn]]
  }
  df
}

labels_to_colnames = function(df) {
  cn = sapply(df, attr, "label")
  stopifnot(is.vector(cn) && length(cn) == ncol(df))
  cn = janitor::make_clean_names(cn)
  colnames(df) = cn
  df
}

# apply the variable names as the label so that we can use View()
# to see the original variable names
remake_labels = function(df) {
  for (icol in colnames(df)) {
    x = df[[icol]]
    label = attr(x, "label")
    variable_name = attr(x, "variable_name")
    attr(x, "label") = variable_name
    attr(x, "full_label") = label
    df[[icol]] = x
  }
  df
}

read_and_relabel = function(
    table, ...,
    folder = "demographics",
    recode_vars = NULL,
    nchar = 100
) {
  file = here::here("data", folder, paste0(table, ".XPT"))
  if (!file.exists(file)) {
    url = nhanes_xpt_url(table)
    curl::curl_download(url, file)
  }
  df = haven::read_xpt(file, ...)
  message(paste0("Read in ", basename(file)))
  table = normalize_table_name(table)
  nh_table = nh_table_name(table)
  translations = nhanesA::nhanesTranslate(
    nh_table = nh_table, data = NULL,
    colnames = colnames(df),
    nchar = nchar)
  message(paste0("Translated ", basename(file)))

  df = nhanesA::nhanesTranslate(nh_table = nh_table, data = df,
                                colnames = colnames(df),
                                nchar = nchar)
  out = recode_demog(
    table = table,
    df = df,
    translations = translations,
    nchar = nchar)
  df = attach_translations(df, translations = translations)
  df = labels_to_colnames(df)

  norm_table = normalize_table_name(nh_table)
  df = process_demog(df)
  df = df %>%
    dplyr::mutate(
      nh_table = table,
      table = norm_table,
      wave = get_wave(nh_table),
      version = paste0("pax_", wave)
    )
  df = remake_labels(df)
  df = tibble::as_tibble(df)
  df
}
waves = c("DEMO_G", "DEMO_H", "DEMO_Y")
dfs = lapply(waves[1], read_and_relabel)
names(dfs) = waves
label_df = function(df) {
  tibble::tibble(
    new = colnames(df),
    original = sapply(df, function(x) {
      attr(x, "variable_name") %||% NA_character_
    })
  )
}
get_col_from_label = function(df, label) {
  labels = sapply(df, attr, "label")
  df[, labels %in% label]
}
dplyr::bind_rows(dfs)
label_df(dfs$DEMO_Y) %>% filter(new == "total_number_of_people_in_the_household")
