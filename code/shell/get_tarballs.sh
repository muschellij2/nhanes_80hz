#!/bin/bash
# id=83730
if [ -z ${version} ];
then
  version=pax_h
fi

raw_dir=data/raw
outdir=${raw_dir}/${version}
echo "outdir is ${outdir}"
mkdir -p ${outdir}
# index
id_file=${raw_dir}/${version}_ids.txt
ids=`cat ${id_file}`
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


curl --remote-name-all https://ftp.cdc.gov/pub/${version}/{73557,73558,73559,73560,73561}.tar.bz2

