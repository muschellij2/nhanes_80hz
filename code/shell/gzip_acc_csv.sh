#!/bin/bash
for i in data/test_acc_csv/pax_y/csv/*.csv;
do
  echo "${i}"
  gzip -9 -f ${i};
done
