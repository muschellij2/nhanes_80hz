library(haven)
library(dplyr)
library(nhanesA)
library(rlang)
source("code/R/utils.R")
table = "DEMO_G"

process_demog = function(df) {
  if ("full_sample_2_year_mec_exam_weight" %in% colnames(df)) {
    atr = attributes(df$full_sample_2_year_mec_exam_weight)
    df = df %>%
      dplyr::mutate(
        full_sample_2_year_mec_exam_weight =
          dplyr::if_else(full_sample_2_year_mec_exam_weight == 0,
                         NA_real_,
                         full_sample_2_year_mec_exam_weight)
      )
    attributes(df$full_sample_2_year_mec_exam_weight) = atr
  }
  if ("age_in_years_at_screening" %in% colnames(df) &&
      is.numeric(df$age_in_years_at_screening)) {
    atr = attributes(df$age_in_years_at_screening)
    df = df %>%
      dplyr::mutate(
        age_in_years_at_screening =
          ifelse(age_in_years_at_screening >= 80,
                 "80 years of age and over",
                 age_in_years_at_screening),
        age_in_years_at_screening = factor(
          age_in_years_at_screening,
          levels = c(0:79,  "80 years of age and over"))
      )
    attributes(df$age_in_years_at_screening) = atr
  }
  df
}

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

read_and_relabel = function(table, ...) {
  file = here::here("data", "demographics", paste0(table, ".XPT"))
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
    nchar = 100)
  message(paste0("Translated ", basename(file)))

  df = nhanesA::nhanesTranslate(nh_table = nh_table, data = df,
                                colnames = colnames(df),
                                nchar = 100)
  recode_vars = NULL
  if (table == "DEMO_G") {
    recode_vars = c("DMDHHSIZ", "DMDFMSIZ", "DMDHHSZA",
                    "DMDHHSZB", "DMDHHSZE", "RIDAGEYR")
  } else if (table == "DEMO_Y") {
    recode_vars = c("DMDHHSIZ")
  }
  if (!is.null(recode_vars)) {
    # cross coding!
    df = nhanesA::nhanesTranslate(
      nh_table = "DEMO_H",
      data = df,
      colnames = recode_vars,
      nchar = 100)
    translations[names(translations) %in% recode_vars] = NULL
    tt = nhanesA::nhanesTranslate(
      nh_table = "DEMO_H",
      data = df,
      colnames = recode_vars,
      nchar = 100)
    translations = c(translations, tt)
  }

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
  df = tibble::as_tibble(df)
  df
}
waves = c("DEMO_G", "DEMO_H", "DEMO_Y")
dfs = lapply(waves, read_and_relabel)
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
