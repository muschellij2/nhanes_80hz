#!/bin/bash

#SBATCH --job-name=parallel_job
#SBATCH --array=1-100 # Replace with your desired number of tasks
#SBATCH --cpus-per-task=1 # Adjust as needed
#SBATCH --time=00:10:00 # Adjust as needed

source ~/.bash_profile || true
conda activate accelerometer

# Run the job
accProcess 62467.csv.gz --timeZone UTC --csvTimeFormat 'yyyy-MM-dd HH:mm:ss.SSS' --csvTimeXYZTempColsIndex 0,1,2,3 --sampleRate 80 --deleteIntermediateFiles False --outputFolder ./test

