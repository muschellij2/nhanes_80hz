
<!-- README.md is generated from README.Rmd. Please edit that file -->

# nhanes_80hz

<!-- badges: start -->
<!-- badges: end -->

The goal of `nhanes_80hz` is to process the 80Hz raw data from the
[NHANES](https://wwwn.cdc.gov/nchs/nhanes/Search/DataPage.aspx?Component=Examination)
accelerometers. Namely the `PAX_H`, `PAX_G` from NHANES, and `PAX_Y`
from [NHANES National Youth Fitness
Survey](https://www.cdc.gov/nchs/nnyfs/index.htm) (NNYFS) study waves.

# Notes

Although the waves of each study are indexed by a letter (`G`, `H`,
`Y`), the folders were named with a prefix of `pax`. In NHANES, the
`pax` stands for physical activity. Although not all data in here is
physical activity, such as demographics, the folder names `pax_g`,
`pax_h`, and `pax_y` are used consistently in subfolders of data to keep
the process consistent.

The NNYFS format is different than NHANES in that most waves of NHANES
use `TABLE_WAVE` format, the NNYFS uses `WAVE_TABLE`. For example, if
the table was `DEMO` for demographics, the table from the CDC and file
for the data (a SAS XPORT file `.XPT`) would be indexed `DEMO_G` for
NHANES but `Y_DEMO` for NNYFS. We have standardized the NNYFS data to
the `TABLE_WAVE` format.

# Code for 80Hz data

Almost all folders are structured in data such that it is:
`data/TYPE/WAVE/ID.ext` where `TYPE` references the type of data, `WAVE`
refers to `pax_g`, `pax_h`, and `pax_y`, and `ID` is the `SEQN` number
of the participant, and `ext` is the extension (usually `csv.gz`,
`tar.bz2`, or `rds`).

For processing data, the order of operations is:

1.  `code/R/get_filenames.R` to extract the links to the tarballs from
    the website HTML for pages such as <https://ftp.cdc.gov/pub/pax_h/>.
    This will create files in `data/raw`, namely `all_filenames.rds`,
    which contains all the required paths to run all the processing.
2.  `code/R/get_tarballs.R` will download the tarballs (`.tar.bz2`) from
    all the studies into `data/raw`, separated by study wave.
3.  `code/R/tarball_to_csv.R` will extract the tarball, concatenate all
    CSVs relevant to physical activity and create outputs in `data/csv`
    as well as any CSV log files (`data/logs`), which includes flags),
    and a metadata file (`data/meta`) with the contents of the tarball.
    This calls the `tarball_df` function in \`code/R/helper_functions.R.
    At this stage, the 80Hz data is in CSV format.
4.  `code/R/resample_data.R` will resample the 80Hz CSV data into
    different sample rates, including downsampling to 30Hz and
    upsampling to 100Hz, into a series of folders named by the sample
    rate, such as `csv_30` and `csv_100`, respectively. This uses
    `walking::resample_accel_data`.
5.  `code/R/get_1s_counts.R` creates ActiGraph Activity Counts (AC)
    using the `agcounter` package.
6.  `code/R/write_steps_data.R` takes the output from ActiLife (in
    `data/acc_steps_1s`) and reformats the data (without the ActiLife
    header) and outputs the 1-second steps data into the `data/steps_1s`
    folder.
7.  `code/R/run_oak_verisense.R` runs the
    [oak](https://github.com/onnela-lab/forest/) and
    [verisense](https://github.com/ShimmerEngineering/Verisense-Toolbox/tree/master/Verisense_step_algorithm)
    algorithm for walking detection and step estimation, implemented in
    the `walking` R package.

## Additional data

8.  `code/R/write_acc_csv.R` will write out the 80Hz data in CSV files
    to the `data/acc_csv` folder that are in the format for ActiLife
    software to ingest, using `write.gt3x::write_actigraph_csv`. The
    output of these from ActiLife are in `data/acc_steps_1s`.
9.  `code/R/download_xpts.R` downloads the SAS XPORT files (.XPT) for
    demographics, physical activity at the minute and day level (MIMS
    units).
10. `code/R/split_daily_minutely.R` splits the XPT files for daily and
    minute-level physical activity data and creates individua CSV files
    to harmonize the data in the same format as the rest of the data and
    allows users to download data from specific participants versus the
    whole cohort.

# Non-CRAN Packages used

1.  `walking`: <https://github.com/muschellij2/walking>
2.  `SummarizedActigraphy`:
    <https://github.com/muschellij2/SummarizedActigraphy>
3.  `write.gt3x`: <https://github.com/muschellij2/write.gt3x>
4.  `stepcount`: <https://github.com/jhuwit/stepcount>, wraps
    <https://github.com/OxWearables/stepcount>
5.  `agcounter`: create ActiLife/AcitGraph Activity Counts
    <https://github.com/muschellij2/agcounter>. The packages
    [`agcounts`](https://github.com/bhelsel/agcounts) and
    [`actilifecounts`](https://github.com/jhmigueles/actilifecounts)
    have different implementations, all aiming at reproducing counts as
    laid out in <https://github.com/actigraph/agcounts>.

## `stepcount`

Version of `stepcount` used was 3.2.4. Please see
`nhanes_80hz_stepcount.yml` for the YAML to create the `stepcount` conda
environment:

``` bash
conda create -f nhanes_80hz_stepcount.yml
```
