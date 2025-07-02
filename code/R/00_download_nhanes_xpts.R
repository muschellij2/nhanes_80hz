library(rvest)
library(dplyr)
library(curl)
library(here)
library(purrr)

outdir = here::here("data/nhanes")
fs::dir_create(outdir)
components = c("Demographics", "Dietary", "Examination", "Laboratory", "Questionnaire")
cycles = c(2011, 2013)
urls = expand.grid(Component = components, CycleBeginYear = cycles)
urls = apply(urls, 1, function(x) {
  paste0("https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=", x[1],
         "&CycleBeginYear=", x[2])
})
url = urls[1]
doc = read_html(url)

# get the urls for the data files
all_urls = map(urls, function(the_url) {
  doc = read_html(the_url)
  href = doc %>%
    html_nodes("#GridView1 a")
  urls = html_attr(href, "href")
  documentation = urls[grepl("[.]htm(l|)$", ignore.case = TRUE, urls)]
  xpt = urls[grepl("[.]xpt(l|)$", ignore.case = TRUE, urls)]
  list(
    documentation = documentation,
    xpt = xpt
  )
})

base_url = "https://wwwn.cdc.gov"

download_urls = map(all_urls, function(x) {
  x$xpt = paste0(base_url, x$xpt)
  x$documentation = paste0(base_url, x$documentation)
  x = unname(unlist(x))
  x
})

download_urls = unname(unlist(download_urls))

purrr::map(download_urls, function(the_url) {
  outfile = file.path(outdir, basename(the_url))
  if (!file.exists(outfile)) {
    x = try({curl::curl_download(the_url, outfile)})
    if (inherits(x, "try-error")) {
      print(the_url)
      # fname = basename(the_url)
      # the_url = paste0("https://wwwn.cdc.gov/Nchs/Nnyfs/", fname)
      # x = try({curl::curl_download(the_url, outfile)})
    }
  }
})
