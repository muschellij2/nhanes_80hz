#!/bin/bash
for i in data/test_acc_csv/pax_h/*.csv.gz;
do
  echo "${i}"
  gunzip ${i};
done
