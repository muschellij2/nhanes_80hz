library(curl)
base_url = "https://wearables-files.ndph.ox.ac.uk/files/models/stepcount/"
model_dir = here::here("stepcount_models")
if (!dir.exists(model_dir)) {
  dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
}
curl::curl_download(
  paste0(base_url, "20230713.joblib.lzma"),
  here::here("stepcount_models/rf-20230713.joblib.lzma"))
curl::curl_download(
  paste0(base_url, "ssl-20230208.joblib.lzma"),
  here::here("stepcount_models/ssl-20230208.joblib.lzma")
)
