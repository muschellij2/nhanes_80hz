longformat="--format=JobID,JobName,Partition,MaxVMSize,MaxRSS,AveRSS,MinCPU,AveCPU,NTasks,AllocCPUS,Elapsed,State,ExitCode,ReqMem,MaxDiskRead,AveDiskRead,MaxDiskWrite"
Rnosave code/R/get_filenames.R -J FILENAMES

# Getting the files/tarballs
# Rnosave code/R/get_files.R -N GET_FILES -l mem_free=2G,h_vmem=3G -t 1-200
Rnosave code/R/get_files.R -J GET_FILES --array=1-200 --mem=3G

# Converting the tarball
# Rnosave code/R/tarball_to_csv.R -N TARBALL -t 1-200 -l mem_free=20G,h_vmem=21G
# Rnosave code/R/tarball_to_csv.R -N TARBALL --array=1-200 --mem=21G

Rnosave code/R/copy_csv_with_new_header.R -J COPIER --array=1-200 --mem=10G -o %x_%A_%a.out -e %x_%A_%a.err


# Rnosave code/R/run_process.R -N PROC -hold_jid_ad TARBALL -t 1-200 -l mem_free=20G,h_vmem=21G
# dep=`get_job_id TARBALL`
# if [[ -n "${dep}" ]]; then
#   dependency="--dependency=aftercorr:$dep"
# else
#   dependency=
# fi
# Rnosave code/R/tarball_to_csv.R -N TAR --array=1-200 --mem=21G ${dependency}
Rnosave code/R/resample_data.R -J RESAMPLE --array=1-200  --mem=22G ${dependency}
Rnosave code/R/check_output_csv.R -J CHECK --array=1-200 -o %x_%A_%a.out -e %x_%A_%a.err


onecpu="--nodes=1 --ntasks=1 --cpus-per-task=1"
Rnosave code/R/get_1s_counts.R -N ONESEC -t 1-5 -l mem_free=20G,h_vmem=21G
Rnosave code/R/get_1s_counts.R -J ONESEC --array=1-200 "${onecpu}" --mem=20G -o %x_%A_%a.out -e %x_%A_%a.err

onecpu="--nodes=1 --ntasks=1 --cpus-per-task=1"
Rnosave code/R/write_acc_csv.R -J ACC_CSV --array=51-52,54-55,57-64,66-67,69-76,78-137 "${onecpu}" --mem=10G -o %x_%A_%a.out -e %x_%A_%a.err
# --exclude=compute-115
Rnosave code/R/write_acc_csv.R -J ACC_CSV --array=18 "${onecpu}" --mem=10G -o %x_%A_%a.out -e %x_%A_%a.err
Rnosave code/R/write_acc_csv.R -N ACC_CSV -t 18 -l mem_free=12G,h_vmem=12G



Rnosave code/R/05_output_get_counts_from30Hz.R -J COUNTS --array=8,18,39,110 --mem=22G -o %x_%A_%a.out -e %x_%A_%a.err


# Rnosave code/R/resample_data.R -N RESAMPLE -t 1-200 -l mem_free=22G,h_vmem=22G
Rnosave code/R/resample_data.R -J RESAMPLE --array=1-100 --mem=22G -o %x_%A_%a.out -e %x_%A_%a.err
Rnosave code/R/05_output_run_process.R -J PROC --array=1-200 --mem=40G -o %x_%A_%a.out -e %x_%A_%a.err

# Rnosave code/R/resample_data.R -J RESAMPLE --array=1-200 --mem=30G -o %x_%A_%a.out -e %x_%A_%a.err

Rnosave code/R/split_daily_minutely.R -N SPLIT -l mem_free=20G,h_vmem=21G
Rnosave code/R/split_daily_minutely.R -J SPLIT --mem=21G

Rnosave code/R/run_oak_verisense.R -N WALK -t 119 -l mem_free=101G,h_vmem=102G
Rnosave code/R/run_oak_verisense.R -J WALK --array=101-200 --mem=52G -o %x_%A_%a.out -e %x_%A_%a.err


Rnosave code/R/write_steps_data.R -J STEPS --mem=20G -o %x_%A.out -e %x_%A.err

Rnosave code/R/flag_all_zero.R -J ZERO --mem=8G -o %x_%A.out -e %x_%A.err --time=4-00:00:00

# NAME_TASKID_JOBID may be best
Rnosave code/R/run_stepcount.R -J STEPCOUNT --array=32,49,66,71,102 --mem=50G -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00
Rnosave code/R/run_stepcount.R -J STEPCOUNT --array=187 --mem=35G -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00

Rnosave code/R/run_nonwear_weartime.R -J WEARTIME --array=40,73,156 --mem=20G -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00
Rnosave code/R/run_nonwear_swan.R -J SWAN --array=8,18,39,110 --mem=20G -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00
# Rnosave code/R/run_nonwear_swan.R -J SWAN --array=1-64,66,74,84,88,89,92,93,110,118 --mem=20G -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00

Rnosave code/R/run_ac_nonwear_sleep.R -J SLEEP --array=1-200 --mem=10G -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00



Rnosave code/R/run_calibration.R -J CALIBRATE --array=1-200 --mem=35G -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00
Rnosave code/R/run_calibration.R -J CALIBRATE --array=1,2,4-9 --mem=35G -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00

Rnosave code/R/05_output_run_mims.R -J MIMS --array=12,161 --mem=40G  -o %x_%A_%a.out -e %x_%A_%a.err
Rnosave code/R/10_mims_comparison.R -J CHECKMIMS --mem=10G  -o %x_%A.out -e %x_%A.err


Rnosave code/R/run_GGIR.R -J GGIR --array=1-200 -o %x_%A_%a.out -e %x_%A_%a.err --time=4-00:00:00


# Rnosave data/lily/code/run_adept.R -J ADEPT --nodes=1 --ntasks=1 --cpus-per-task=8  --array=11-20 --mem=100G -o %x_%A_%a.out -e %x_%A_%a.err
Rnosave data/lily/code/run_adept_byrank.R -J ADEPT --nodes=1 --ntasks=1 --cpus-per-task=8  --array=11-20 --mem=100G -o %x_%A_%a.out -e %x_%A_%a.err

sbatch code/shell/run_accProcess.sh
# sbatch --array=1-130 code/shell/run_accProcess.sh

# --exclude=compute-115
# accProcess 62161.csv --csvTimeFormat 'yyyy-MM-dd HH:mm:ss.SSS' --csvTimeXYZTempColsIndex 0,1,2,3 --sampleRate 80
# if we want wear time
accProcess 62467.csv.gz --timeZone UTC --csvTimeFormat 'yyyy-MM-dd HH:mm:ss.SSS' --csvTimeXYZTempColsIndex 0,1,2,3 --sampleRate 80 --deleteIntermediateFiles False --outputFolder ./test
# --deleteIntermediateFiles False
# --outputFolder ./test


Rnosave code/R/999_check_min_times.R -J CHECK --array=1-200 --mem=10G -o %x_%A_%a.out -e %x_%A_%a.err

