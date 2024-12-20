#!/bin/bash

#SBATCH --job-name=accProcess
#SBATCH --array=1-200 # Replace with your desired number of tasks
#SBATCH --cpus-per-task=1 # Adjust as needed
#SBATCH --time=4-00:00:00  # time
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH -o "%x_%A_%a.out"
#SBATCH -e "%x_%A_%a.err"

source ~/.bash_profile | true  # Load Python if needed
echo "before activate conda env is $CONDA_DEFAULT_ENV"
conda activate accelerometer
echo "conda env is $CONDA_DEFAULT_ENV"
conda activate accelerometer
echo "conda env is $CONDA_DEFAULT_ENV"

tempfile=$(mktemp)
Rscript code/R/cat_csv_files.R  > $tempfile


while read -r line; do
  echo "$line";
  version=`dirname $line`
  version=`basename $version`
  # bn=$(basename $line)
  accProcess ${line} --timeZone UTC \
  --csvTimeFormat 'yyyy-MM-dd HH:mm:ss.SSS' \
  --csvTimeXYZTempColsIndex 0,1,2,3 \
  --verbose True \
  --sampleRate 100 \
  --csvSampleRate 80 \
  --deleteIntermediateFiles False \
  --outputFolder data/accProcess/$version
  # --m10l5 True \
  # --psd True \
  # --fourierFrequency True \
done < $tempfile

