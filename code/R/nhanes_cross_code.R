
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

cross_code_list = list(
  DEMO_G = list(
    recode_vars = c("DMDHHSIZ", "DMDFMSIZ", "DMDHHSZA",
                    "DMDHHSZB", "DMDHHSZE", "RIDAGEYR"),
    recode_nh_table = "DEMO_H"
  ),
  DEMO_Y = list(
    recode_vars = c("DMDHHSIZ"),
    recode_nh_table = "DEMO_H"
  )
)

nhanes_crosscode = function(
    df,
    translations,
    recode_nh_table = "DEMO_H",
    recode_vars = NULL,
    nchar = 100) {
  if (!is.null(recode_vars)) {
    # cross coding!
    df = nhanesA::nhanesTranslate(
      nh_table = recode_nh_table,
      data = df,
      colnames = recode_vars,
      nchar = nchar)
    translations[names(translations) %in% recode_vars] = NULL
    tt = nhanesA::nhanesTranslate(
      nh_table = recode_nh_table,
      data = NULL,
      colnames = recode_vars,
      nchar = nchar)
    translations = c(translations, tt)
  } else {
    warning("recode_vars is NULL")
  }
  list(df = df,
       translations = translations)
}


nhanes_crosscode_data = function(table, df, translations, nchar = 100) {
  stopifnot(table %in% names(cross_code_list))
  out = cross_code_list[[table]]
  recode_nh_table = out$recode_nh_table
  recode_vars = out$recode_vars

  out = nhanes_crosscode(df = df,
                         translations = translations,
                         recode_nh_table = recode_nh_table,
                         recode_vars = recode_vars,
                         nchar = nchar
  )
  out
}



