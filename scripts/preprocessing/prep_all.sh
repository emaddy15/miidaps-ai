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
export NUCAPS_SDR=/data/users/emaddy/cris_fsr/nucaps_data/${dateRun}
export GDAS_DUMP_DIR=${OUTPUTDATA_PATH}/gfs_dump/
export GDAS_DATA_DIR=${OUTPUTDATA_PATH}/gfs_data/

#---run get data and dump data
${PREPROCESSING_PATH}/download_gfs_data.sh 
${PREPROCESSING_PATH}/dump_gfs_data.sh 

export instrConfigFile=${AITUNEDATA_PATH}/InstrConfig_j01_crisfsr.dat
export channelSet=${AITUNEDATA_PATH}/cris_431lwmid.list
export gdas_type=1
export AIFileOut=${OUTPUTDATA_PATH}/aifile_${DATE}

OUTPUTLIST_PATH=${OUTPUTDATA_PATH}/lists
if [ ! -d ${OUTPUTLIST_PATH} ]; then 
  mkdir -p ${OUTPUTLIST_PATH}
fi

NetCDFFile_List=${OUTPUTLIST_PATH}/${DATE}_NUCAPS.list
ls ${NUCAPS_SDR}/*nc > ${NetCDFFile_List}

${PYTHON} ${PYSCRIPT} ${NetCDFFile_List} ${instrConfigFile} ${gdas_type} ${GDAS_DUMP_DIR} ${channelSet} ${AIFileOut} 





