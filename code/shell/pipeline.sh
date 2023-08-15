Rnosave code/R/get_filenames.R -N FILENAMES

# Getting the files/tarballs
Rnosave code/R/get_files.R -N GET_FILES -l mem_free=2G,h_vmem=3G -t 1-56

# Converting the tarball
Rnosave code/R/tarball_to_csv.R -N TARBALL -t 1-56 -l mem_free=20G,h_vmem=21G

Rnosave code/R/run_process.R -N PROC -hold_jid_ad TARBALL -t 1-56 -l mem_free=20G,h_vmem=21G

Rnosave code/R/split_daily_minute.R -N SPLIT -l mem_free=20G,h_vmem=21G
