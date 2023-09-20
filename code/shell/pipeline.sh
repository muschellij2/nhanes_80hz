Rnosave code/R/get_filenames.R -N FILENAMES

# Getting the files/tarballs
# Rnosave code/R/get_files.R -N GET_FILES -l mem_free=2G,h_vmem=3G -t 1-200
Rnosave code/R/get_files.R -N GET_FILES --array=1-200 --mem=3G

# Converting the tarball
Rnosave code/R/tarball_to_csv.R -N TARBALL -t 1-200 -l mem_free=20G,h_vmem=21G
# Rnosave code/R/tarball_to_csv.R -N TARBALL --array=1-200 --mem=21G

Rnosave code/R/run_process.R -N PROC -hold_jid_ad TARBALL -t 1-200 -l mem_free=20G,h_vmem=21G
# dep=`get_job_id TARBALL`
# if [[ -n "${dep}" ]]; then
#   dependency="--dependency=aftercorr:$dep"
# else
#   dependency=
# fi
# Rnosave code/R/tarball_to_csv.R -N PROC --array=1-200 --mem=21G ${dependency}
Rnosave code/R/resample_data.R -J PROC --array=1-200  --mem=22G ${dependency}

Rnosave code/R/resample_data.R -N RESAMPLE -t 5-200 -l mem_free=30G,h_vmem=30G
Rnosave code/R/resample_data.R -J RESAMPLE --array=195-200 --mem=30G -o %x_%A_%a.out -e %x_%A_%a.err
# Rnosave code/R/resample_data.R -J RESAMPLE --array=1-200 --mem=30G -o %x_%A_%a.out -e %x_%A_%a.err

# Rnosave code/R/split_daily_minutely.R -N SPLIT -l mem_free=20G,h_vmem=21G
Rnosave code/R/split_daily_minutely.R -J SPLIT --mem=21G

Rnosave code/R/run_oak_verisense.R -N WALK -t 1-5 -l mem_free=51G,h_vmem=54G
Rnosave code/R/run_oak_verisense.R -J WALK --array=195-200 --mem=52G -o %x_%A_%a.out -e %x_%A_%a.err

