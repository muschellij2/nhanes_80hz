if [ -z "$1" ];
then
  echo "Usage: ./fix_header_rows.sh file.csv.gz"
  exit 1
fi
ifile=${1}

unzipper() {
  x=`which pigz`
  if [ -z "${x}" ];
  then
    echo "gunzip"
  else
    echo "pigz -d"
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
fix_header_rows() {
  outfile=${1}
  if [ ! -f ${outfile} ];
  then
    gsutil -m cp gs://nhanes_80hz/${outfile} ${outfile}
  fi
  the_head=`zcat ${outfile} | head -n 1`
  first_row=`zcat ${outfile} | head -n 2 | tail -n 1`
  echo ${first_row} | grep -q ","
  if [ $? -ne 0 ];
  then
    echo "${outfile} has tab delimited data! Wrong from vroom::write"
    gsutil rm gs://nhanes_80hz/${outfile}
    exit 1
  fi

  # first_row=`zcat ${outfile} | head -n 280001 | grep "HEADER"`
  # echo $first_row | grep "HEADER" |  -gt 1
  # echo ${first_row} | grep -q ","
  # if [ $? -ne 0 ];
  # then
  #   echo "${outfile} has tab delimited data! Wrong from vroom::write"
  #   exit 1
  # fi
  if [ "${the_head}" != "HEADER_TIMESTAMP,X,Y,Z" ];
  then
    unzip_function=`unzipper`
    file=`echo "${outfile}" | sed -e 's/\.gz//'`
    echo "Unzipping file ${unzip_function}"
    ${unzip_function} -f ${outfile}
    sed -i '/HEADER/d' ${file}
    sed -i '1s/^/HEADER_TIMESTAMP,X,Y,Z\n/' ${file}
    zip_function=`zipper`
    # echo "ls id_dir: ${id_tmp_dir}"
    # (ls -U ${id_tmp_dir} | head -n 5) || true
    echo "Doing header resave"
    ${zip_function} -f -9 ${file}
    gsutil -m cp ${outfile} gs://nhanes_80hz/${outfile}
  fi
}


grep -q ${ifile} headers_checked.txt
if [ $? -ne 0 ];
then
  fix_header_rows ${ifile}
  echo "${ifile}" >> headers_checked.txt
fi
rm -f ${ifile}
