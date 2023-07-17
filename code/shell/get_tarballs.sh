# id=83730
if [ -z ${version} ];
then
  version=pax_h
fi

outdir=data/raw/${version}
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
    # echo "ls ${outdir}"
    # ls -al ${outdir}
    if [ ! -f ${outfile} ];
    then
      echo "Downloading File"
      curl -p --insecure  "https://ftp.cdc.gov/pub/${version}/${file}" -o "${outfile}"
    fi
  fi

  rm -f ${outfile}
done



