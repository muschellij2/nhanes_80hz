library(stepcount)
unset_reticulate_python()
use_stepcount_condaenv()
df = readr::read_csv("data/csv_30/pax_h/73557.csv.gz")

model_file =   here::here("stepcount_models/ssl-20230208.joblib.lzma")
model_type = "ssl"


debugonce(stepcount)
out = stepcount(df)
