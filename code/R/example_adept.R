library(dplyr)
library(ggplot2)
library(walking)
library(adept)
# muschellij2/adept@nhanes
library(adeptdata)
options(digits.secs = 3)
source(here::here("code/R/helper_functions.R"))
all_wrist_templates = adeptdata::stride_template$left_wrist
template_list = do.call(rbind, all_wrist_templates)
template_list = apply(template_list, 1, identity, simplify = FALSE)

files = list.files(here::here("data", "csv"),
                   pattern = ".csv.gz", full.names = TRUE, recursive = TRUE)

file = files[1]
print(file)
data = read_80hz(file, progress = FALSE)
sample_rate = 80L

data = standardize_data(data, subset = TRUE)
data15 = walking:::resample_accel_data(data, sample_rate = 15L)


oak = walking::estimate_steps_forest(data = data)
vs = walking::estimate_steps_verisense(data = data, sample_rate = sample_rate,
                                       method = "original")
vs_revised = walking::estimate_steps_verisense(data = data, sample_rate = sample_rate,
                                               method = "revised")


oak15 = walking::estimate_steps_forest(data = data15)
vs15 = walking::estimate_steps_verisense(
  data = data15,
  sample_rate = 15L,
  method = "original")
vs15_revised = walking::estimate_steps_verisense(
  data = data15,
  sample_rate = 15L,
  method = "revised")

# xyz = data %>% select(X, Y, Z) %>% as.matrix()
# rm(data)
# walk_out = segmentWalking(xyz,
#                           xyz.fs = sample_rate,
#                           template = template_list)

