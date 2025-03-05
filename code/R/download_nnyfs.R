library(rvest)
library(dplyr)
library(curl)
library(here)

outdir = here::here("data/nnyfs")
doc = read_html("https://wwwn.cdc.gov/nchs/nhanes/search/nnyfsdata.aspx")

base_url = "https://wwwn.cdc.gov"
href = doc %>%
  html_nodes("#DataFileUrl a")
urls = html_attr(href, "href")
urls = urls[grepl("[.]xpt$", ignore.case = TRUE, urls)]
urls = paste0(base_url, urls)

# sapply(urls, function(the_url) {
#   outfile = file.path(outdir, basename(the_url))
#   if (!file.exists(outfile)) {
#     curl::curl_download(the_url, outfile)
#   }
# })

sapply(urls, function(the_url) {
  fname = basename(the_url)
  url = paste0("https://wwwn.cdc.gov/Nchs/Nnyfs/", fname)
  outfile = file.path(outdir, fname)
  if (!file.exists(outfile)) {
    curl::curl_download(url, outfile)
  }
})

# documentation
href = doc %>%
  html_nodes("#GridView1 a")
urls = html_attr(href, "href")
urls = urls[grepl("[.]htm(l|)$", ignore.case = TRUE, urls)]
urls = paste0(base_url, urls)

purrr::map(urls, function(the_url) {
  outfile = file.path(outdir, basename(the_url))
  if (!file.exists(outfile)) {
    x = try({curl::curl_download(the_url, outfile)})
    if (inherits(x, "try-error")) {
      fname = basename(the_url)
      the_url = paste0("https://wwwn.cdc.gov/Nchs/Nnyfs/", fname)
      x = try({curl::curl_download(the_url, outfile)})
    }
  }
})
