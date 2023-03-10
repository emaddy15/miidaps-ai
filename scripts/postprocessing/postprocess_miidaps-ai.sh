#!/bin/sh 

echo "Running PostProcessor ...."
echo 

DateStart=`date`
echo "Date Start: ${DateStart}"

#---python postprocessing script
PYSCRIPT=${PYTHON_POST_PATH}/postprocess_miidaps-ai.py

#---input data from preprocessing
aiFileAtm=${INPUTDATA_PATH}/aifile_${YYYYMMDD}_j01_atms_j01_cris.atm
modelName=${RUNID}

#---output from miidaps-ai in internal format and on
#   hybrid-sigma pressure grid
aiFilePredict=${RUNDIR}/${modelName}_${YYYYMMDD}.nn
aiFilePredict_CrIS=${RUNDIR}/${modelName}_${YYYYMMDD}.nn.cris
aiFilePredict_ATMS=${RUNDIR}/${modelName}_${YYYYMMDD}.nn.atms

#---output file names for netCDF output -
#   interpolation to fixed 100 pressure layer, 101 pressure level grid
#   simple qc applied
aiFileNetCDF=${RUNDIR}/${modelName}_${YYYYMMDD}.nc
aiFileNetCDF_CrIS=${RUNDIR}/${modelName}_${YYYYMMDD}_cris.nc
aiFileNetCDF_ATMS=${RUNDIR}/${modelName}_${YYYYMMDD}_atms.nc

#---run post-processing converter
stdbuf -oL ${PYTHON} ${PYSCRIPT} ${aiFileAtm} ${aiFilePredict} ${aiFileNetCDF} ${instrConfigFile} ${channelSet}

stdbuf -oL ${PYTHON} ${PYSCRIPT} ${aiFileAtm} ${aiFilePredict_CrIS} ${aiFileNetCDF_CrIS} ${instrConfigFile} ${channelSet}

stdbuf -oL ${PYTHON} ${PYSCRIPT} ${aiFileAtm} ${aiFilePredict_ATMS} ${aiFileNetCDF_ATMS} ${instrConfigFile} ${channelSet}

DateEnd=`date`
echo "Date End: ${DateEnd}"
