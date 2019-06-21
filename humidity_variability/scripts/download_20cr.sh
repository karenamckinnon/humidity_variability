#!/bin/bash

downloaddir=$1
start_yr=$2
end_yr=$3

cwd=$(pwd)
cd $downloaddir

for i in $(seq $start_yr $end_yr)
    do
    wget "ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/Dailies/monolevel/prmsl.$i.nc"
done

cd $cwd
