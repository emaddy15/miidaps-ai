#!/bin/sh

############################
#---setup command and system path information
############################

#---python 
export PYSCRIPT=${PYTHON_PREP_PATH}/prep_nucaps_netcdf.py

#---NUCAPS netcdf data 
#   note that NUCAPS netcdf data should be saved into at least four directories per day around synoptic times
#   00z, 06z; 06z, 12z; 12z, 18z; 18z, 24z or better,
#   -03z, 03z; 03z, 09z; 09z, 15z; 15z, 21z; 21z, +03z
# 
export GDAS_DUMP_DIR=${INPUTDATA_PATH}/gfs_dump/
export GDAS_DATA_DIR=${INPUTDATA_PATH}/gfs_data/

#---run get data and dump data
${PREPROCESSING_PATH}/download_gfs_data.sh 
${PREPROCESSING_PATH}/dump_gfs_data.sh 


