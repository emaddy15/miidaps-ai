#!/bin/sh

echo "Running NUCAPS PreProcessor ...."
echo 

DateStart=`date`
echo "Date Start: ${DateStart}"

if [[ ${PYTHON_VERSION} -eq 3 ]]; then
 export PYSCRIPT=${PYTHON_PREP_PATH}/prep_nucaps_netcdf.py.h5ncdf
else
 export PYSCRIPT=${PYTHON_PREP_PATH}/prep_nucaps_netcdf.py.ncdf4
fi

#---location of NUCAPS co-located CrIS/ATMS files
export NUCAPS_SDR=${NUCAPSDATA_PATH}/${YYYYMMDD}

#---location of GDAS/GFS dump data
export GDAS_DUMP_DIR=${INPUTDATA_PATH}/gfs_dump/
export GDAS_DATA_DIR=${INPUTDATA_PATH}/gfs_data/

export gdas_type=1

#---output file location
export AIFileOut=${INPUTDATA_PATH}/aifile_${YYYYMMDD}


#---make directory for list files
INPUTLIST_PATH=${INPUTDATA_PATH}/lists
if [ ! -d ${INPUTLIST_PATH} ]; then 
  mkdir -p ${INPUTLIST_PATH}
fi

#---get list of NUCAPS files to process
NetCDFFile_List=${INPUTLIST_PATH}/${DATE}_NUCAPS.list
ls ${NUCAPS_SDR}/*nc > ${NetCDFFile_List}

echo "${PYTHON} ${PYSCRIPT} ${NetCDFFile_List} ${instrConfigFile} ${gdas_type} ${GDAS_DUMP_DIR} ${channelSet} ${AIFileOut} "
#export OMP_NUM_THREADS=1

#---run software 
${PYTHON} ${PYSCRIPT} ${NetCDFFile_List} ${instrConfigFile} ${gdas_type} ${GDAS_DUMP_DIR} ${channelSet} ${AIFileOut} 

DateEnd=`date`
echo "Date End: ${DateEnd}"



