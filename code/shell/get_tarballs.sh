# id=83730
if [ -z ${version} ];
then
  version=pax_h
fi

bucket=nhanes_80hz
outdir=${version}/raw
echo "outdir is ${outdir}"
mkdir -p ${outdir}
# index
# sed "${NUM}q;d" file
ids=`cat ${version}_folds.txt | grep " ${fold}$" | awk '{ print $1 }'`
# cat sample_data_1.txt | awk 'NR==25'
echo "ls `pwd`"
ls -al

# head -n 500 ${version}_ids.txt | while read id
# do
for id in ${ids};
do
  echo ${id}
   # do something with $line here
  file=${id}.tar.bz2
  outfile=${outdir}/${file}
  bucket_file=gs://${bucket}/${outfile}
  echo "running gsutil stat ${bucket_file}"
  # you need the -q!!!!!
  gsutil -q stat ${bucket_file};
  if [ $? -ne  0 ];
  then
    # echo "ls ${outdir}"
    # ls -al ${outdir}
    if [ ! -f ${outfile} ];
    then
      echo "Downloading File"
      curl -p --insecure  "https://ftp.cdc.gov/pub/${version}/${file}" -o "${outfile}"
    fi

    echo "Uploading File `pwd`/${outfile} gs://${bucket}/${outfile}"
    gsutil -m cp `pwd`/${outfile} ${bucket_file}
    # echo "ls ${outdir}"
    # ls -al ${outdir}
    # gsutil -m rsync ${outdir} gs://${bucket}/${outdir}
  fi

  rm -f ${outfile}
done



