library(curl)
library(dplyr)
library(stringr)
library(lubridate)

download_80hz = function(id, version, exdir = tempdir(), ...) {
  files = id
  tarball_ending = grepl("[.]tar[.]bz2", id)
  files[!tarball_ending] = paste0(files[!tarball_ending], ".tar.bz2")
  urls = paste0("https://ftp.cdc.gov/pub/", version, "/", files)
  outfiles = sapply(urls, function(x) {
    destfile = file.path(exdir, basename(x))
    if (!file.exists(destfile)) {
      curl::curl_download(x, destfile = destfile, ...)
    }
  })
  outfiles
}

bad_ids = c(63347, 65108)

#' # 65108 Tarball issue
#' Downloading the data
raw = download_80hz("65108", "pax_g")
files = untar(raw, list = TRUE, verbose = FALSE,
              exdir = ".")
log_file = files[grepl("_log", files, ignore.case = TRUE)]
#' Getting only the data files
csv_files = files[!grepl("_log", files, ignore.case = TRUE)]
#' Extracting date
dates = csv_files %>%
  stringr::str_replace(".*(\\d{4}-\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{2}).*", "\\1")
dates = lubridate::ymd_hms(dates)
#' taking difference in dates, in minutes
diff_dates = c(0, diff(dates))
index = which(diff_dates > 60)
#' We can see that 2000-01-12-19 and 2000-01-12-20 are missing
csv_files[seq(index-1, index)]


#' # 63347 Tarball issue
#' Downloading the data
raw = download_80hz("63347", "pax_g")
files = untar(raw, list = TRUE, verbose = FALSE,
              exdir = ".")
log_file = files[grepl("_log", files, ignore.case = TRUE)]
#' Getting only the data files
csv_files = files[!grepl("_log", files, ignore.case = TRUE)]
#' Extracting date
dates = csv_files %>%
  stringr::str_replace(".*(\\d{4}-\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{2}).*", "\\1")
dates = lubridate::ymd_hms(dates)
#' taking difference in dates, in minutes
diff_dates = c(0, diff(dates))
index = which(diff_dates > 60)
#' We can see that 2000-01-07-16-00 is missing
csv_files[seq(index-1, index)]

#' Showing that the data does exist
#' We are going to download the hourly data to see if there are any missing
#' elements from the hourly data
url = "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/PAXHR_G.XPT"
destfile = basename(url)
if (!file.exists(destfile)) {
  curl::curl_download(url, destfile = destfile)
}
hr = haven::read_xpt(url)
#' Keep only the ids we need from above
sub_hr = hr %>%
  filter(SEQN %in% bad_ids)
#' We see 193 records (meaning hours shouldn't be missing)
sub_hr %>% count(SEQN)
#' Looking at what may potentially cause this
sub_hr = sub_hr %>%
  mutate(time = sub_hr$PAXSSNHP/80L/60L) %>%
  dplyr::group_by(SEQN) %>%
  mutate(dtime = c(0, diff(time)))
#' Here we look for MIMS being zero/negative, the difference in time being off
#' (indicating a skip in time), or the number of valid minutes being low
sub_hr %>%
  filter(dtime > 60 | is.na(PAXMTSH) | PAXMTSH <= 0 | PAXVMH < 5) %>%
  knitr::kable()
sub_hr %>%
  filter(PAXAISMH >= (60*80*60) - 5) %>%
  knitr::kable()
