# /bin/bash
get_tmp_dir() {
  tdir=${TMP}
  if [ -z "${tdir}" ];
  then
    if [ -d "/workspace" ];
    then
      tdir=/workspace/tmp
    else
      tdir=/tmp
    fi
  fi
  mkdir -p ${tdir}
  echo "${tdir}"
}

head_tail() {
  file=${1}
  outfile=${2}
  zcat ${file} | head -n 2  > ${outfile}
  zcat ${file} | tail -n 1 >> ${outfile}
}

cat_together() {
  outfile=${1}
  shift
  files=$@
  first=`echo ${files} | awk '{ print $1 }'`
  head -n 1 ${first} > ${outfile}
  echo "header printed to ${outfile}"
  # for i in ${files};
  # do
  #   sed -i 1d ${i}
  # done
  # echo "Done sed"
  # cat ${files} >> ${outfile}
  # echo "Done cat"
  echo "Running tail"
  # for i in ${files};
  # do
  # # echo "${i}"
  # tail +2 ${i} >> ${outfile}
  # done
  tail -q -n +2 ${files} >> ${outfile}
  echo "Done tail"
}

unzipper() {
  x=`which pigz`
  if [ -z "${x}" ];
  then
    echo "gunzip"
  else
    echo "pigz -d"
  fi
}

remove_header_rows() {
  outfile=${1}
  file=`echo "${outfile}" | sed -e 's/\.gz//'`
  unzip_function=`unzipper`
  echo "Unzipping file ${unzip_function}"
  ${unzip_function} ${outfile}
  sed -i '/HEADER/d' ${file}
  sed -i '1s/^/HEADER_TIMESTAMP,X,Y,Z\n/' ${file}
  zip_function=`zipper`
  # echo "ls id_dir: ${id_tmp_dir}"
  # (ls -U ${id_tmp_dir} | head -n 5) || true
  echo "Doing header resave"
  ${zip_function} -f -9 ${file}
}

bz2() {
  x=`which lbzip2`
  if [ -z "${x}" ];
  then
    echo "bzip2"
  else
    echo "lbzip2"
  fi
}

zipper() {
  x=`which pigz`
  if [ -z "${x}" ];
  then
    echo "gzip"
  else
    echo "pigz"
  fi
}

untar_file() {
  tarball=${1}
  id=${2}
  tdir=`get_tmp_dir`
  id_tmp_dir=${tdir}/${id}
  zip_program=`bz2`
  echo "Untarring with ${zip_program}"
  tar xf ${tarball} --use-compress-program=${zip_program} --directory ${id_tmp_dir}
  if [ $? -ne 0 ]; then
    rm -f ${tarball};
    # this gets picked up I believe
    exit $?
  fi
  echo "ls id_tmp_dir: ${id_tmp_dir}"
  ls -l ${id_tmp_dir}
}
make_log_csv() {
  id=${1}

  tdir=`get_tmp_dir`
  id_tmp_dir=${tdir}/${id}
  logs_gzfile=${version}/logs/${id}.csv.gz

  zip_function=`zipper`
  # echo "ls id_dir: ${id_tmp_dir}"
  # (ls -U ${id_tmp_dir} | head -n 5) || true
  echo "Doing Log"
  ${zip_function} -f -9 ${id_tmp_dir}/*Logs.csv
  cp ${id_tmp_dir}/*Logs.csv.gz  ${logs_gzfile} || true
  rm -rf ${id_tmp_dir}/*Logs.csv*
}
make_csv() {
  id=${1}

  tdir=`get_tmp_dir`
  id_tmp_dir=${tdir}/${id}
  nogz_csv_file=${version}/csv/${id}.csv

  zip_function=`zipper`

  txtfile=${tdir}/${id}.txt
  echo "txtfile is ${txtfile}"
  ls ${id_tmp_dir}/*sensor* > ${txtfile}
  echo "wc is `wc -l ${txtfile}`"
  echo "Catting together"
  files=`cat ${txtfile}`

  # tmpfile=${tdir}/data_file_${id}.csv
  # echo "tmpfile is $tmpfile"
  # result=`cat_together ${tmpfile} ${files}`
  # echo "moving to ${nogz_csv_file}"
  # mv ${tmpfile} ${nogz_csv_file}
  first=`echo ${files} | awk '{ print $1 }'`
  the_head=`head -n 1 ${first}`
  if [ "${the_head}" != "HEADER_TIMESTAMP,X,Y,Z" ];
  then
    echo "the head is ${the_head}"
    echo "Header is off!"
    exit 1
  fi

  cat_together ${nogz_csv_file} ${files}
  if [ $? -ne 0 ];
  then
    echo "cat_together failed!"
    # rm -f ${nogz_csv_file};
    # exit 0;
  fi
  echo "Zipping ${nogz_csv_file} with ${zip_function}"
  ${zip_function} -f -9 ${nogz_csv_file}
  if [ $? -ne 0 ];
  then
    rm -f ${nogz_csv_file} ${nogz_csv_file}.gz
  fi
}

if [ -z ${version} ];
then
  version=pax_h
fi

bucket=nhanes_80hz

# Make local files
outdir=${version}/raw
mkdir -p ${outdir}
csv_outdir=${version}/csv
mkdir -p ${csv_outdir}
logs_outdir=${version}/logs
mkdir -p ${logs_outdir}

tdir=`get_tmp_dir`

# Get ids from fold
if [ -z "${ids}" ];
then
  ids=`cat ${version}_folds.txt | grep " ${fold}$" | awk '{ print $1 }'`
fi
echo "ls `pwd`"
ls -al

# head -n 500 ${version}_ids.txt | while read id
# do
echo "ids are ${ids}"
for id in ${ids};
do
  echo ${id}
  tarball=${outdir}/${id}.tar.bz2

  id_tmp_dir=${tdir}/${id}
  csv_gzfile=${csv_outdir}/${id}.csv.gz
  csv_gzfile_bucket=gs://${bucket}/${csv_gzfile}
  logs_gzfile=${logs_outdir}/${id}.csv.gz
  logs_gzfile_bucket=gs://${bucket}/${logs_gzfile}

  echo "running gsutil stat ${csv_gzfile_bucket}"
  gsutil -q stat ${csv_gzfile_bucket};
  bucket_csv_exists=$?
  echo "running gsutil stat ${logs_gzfile_bucket}"
  gsutil -q stat ${logs_gzfile_bucket};
  bucket_log_exists=$?
  if [[ ! ${bucket_csv_exists} -ne 0 ]];
  then
    echo "Downloading File ${csv_gzfile_bucket} ${csv_gzfile}"
    gsutil -m cp ${csv_gzfile_bucket} ${csv_gzfile}
    remove_header_rows ${csv_gzfile}
    echo "Uploading File ${csv_gzfile} ${csv_gzfile_bucket}"
    gsutil -m cp ${csv_gzfile} ${csv_gzfile_bucket}
  fi
  if [[ ${bucket_log_exists} -ne  0 ]] || [[ ${bucket_csv_exists} -ne 0 ]];
  then
    bucket_tarball=gs://${bucket}/${tarball}
    echo "running gsutil stat ${bucket_tarball}"
    # you need the -q!!!!!
    if [ ! -f ${tarball} ];
    then
      gsutil -q stat ${bucket_tarball};
      if [ $? -ne  0 ];
      then
          echo "Downloading File"
          curl -p --insecure  "https://ftp.cdc.gov/pub/${version}/${file}" -o "${tarball}"
      else
          echo "Downloading File GCS"
          gsutil -m cp ${bucket_tarball} `pwd`/${tarball}
      fi
    fi

    ######################################################
    # Creating CSV collapsed
    ######################################################
    if [[ ! -f ${csv_gzfile} ]] || [[ ! -f ${logs_gzfile} ]];
    then
      echo "Ready to Untar"
      mkdir -p ${id_tmp_dir}
      untar_file ${tarball} ${id}
      if [ $? -ne 0 ]; then
        # exit 0
        continue
      fi
      echo "Making Log CSV"
      make_log_csv ${id}
      if [[ -f ${logs_gzfile} ]] && [[ ${bucket_log_exists} -ne 0 ]];
      then
        echo "Uploading File `pwd`/${logs_gzfile} ${logs_gzfile_bucket}"
        gsutil -m cp `pwd`/${logs_gzfile} ${logs_gzfile_bucket}
      fi
      echo "Making CSV"
      make_csv ${id}
      if [[ -f ${csv_gzfile} ]] && [[ ${bucket_csv_exists} -ne 0 ]];
      then
        echo "Uploading File `pwd`/${csv_gzfile} ${csv_gzfile_bucket}"
        gsutil -m cp `pwd`/${csv_gzfile} ${csv_gzfile_bucket}
      fi
    fi
    rm -rf ${id_tmp_dir}
    rm -f ${tarball}
  fi
  rm -rf ${id_tmp_dir}
  rm -f ${tarball}
  rm -f ${csv_gzfile}
done



