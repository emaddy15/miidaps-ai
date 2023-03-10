#!/bin/sh 

##################
YYYY=$1; MM=$2; DD=$3

umask 0022
##################
# Setup paths and instrument related settings (coefficient directories, runid, ...)
# 
source ../../setup/paths.setup.s4
#source ../setup/instrument.setup

##################
#---build date strings
export YYYYMMDD=${YYYY}${MM}${DD}
export DATE=${YYYY}-${MM}-${DD}


##############################################
#---make i/o directories if they don't exist
# inputdata  :: where prepped MiRS data lives
# outputdata :: where all AI algorithm output will go
export INPUTDATA_PATH=${INPUTDATA_PATH}/${INSTRUMENT}/${DATE}
if [ ! -d ${INPUTDATA_PATH} ]; then 
  mkdir -p ${INPUTDATA_PATH}
fi 

export OUTPUTDATA_PATH=${OUTPUTDATA_PATH}/${INSTRUMENT}/${DATE}
if [ ! -d ${OUTPUTDATA_PATH} ]; then 
  mkdir -p ${OUTPUTDATA_PATH}
fi 

#--- make log directory if doesn't exist
export LOGDIR=${OUTPUTDATA_PATH}/logfiles
mkdir -p ${LOGDIR}
export RUNDIR=${OUTPUTDATA_PATH}/run/
mkdir -p ${RUNDIR}

###################
#---user defined runid for output files
export RUNID=v001_431lwmid

#---logfile for all steps
export PROC_CNTLLOG=${LOGDIR}/all_${RUNID}.log
#---remove the process log file
rm -rf ${PROC_CNTLLOG}
touch ${PROC_CNTLLOG}

export AIMODELDATA_PATH=${AIMODELDATA_PATH}/${RUNID}/
export instrConfigFile=${AITUNEDATA_PATH}/InstrConfig_j01_crisfsr.dat
export channelSet=${AITUNEDATA_PATH}/cris_${RUNID}.list

##############################################
#---Settings for run

#---download and dump GDAS/GFS forecasts
prep_gdas=False
prep_gdas=True

#---convert NUCAPS netcdf to AI input 
prep_nucaps=True
prep_nucaps=False

#---run MIIDAPS-AI
run_inference=True
run_inference=False

#---convert MIIDAPS-AI from hybrid-sigma to fixed pressure grid
postprocess_output=True
postprocess_output=False

#---plot daily products 
plot_daily_products=False

##############################################
#---initialize processing time and write flags into log file
echo "AI PROCESSING STARTED FOR ${YYYYMMDD} ---- `date`" >> ${PROC_CNTLLOG}
echo "AI runid=${RUNID} set: " >> ${PROC_CNTLLOG}
echo " prep_gdas           = ${prep_gdas}"  >> ${PROC_CNTLLOG}
echo " prep_nucaps         = ${prep_nucaps}"  >> ${PROC_CNTLLOG}
echo " inference           = ${run_inference}"  >> ${PROC_CNTLLOG}
echo " postprocess_output  = ${postprocess_output}"  >> ${PROC_CNTLLOG}
echo " plot_daily_products = ${plot_daily_products}"  >> ${PROC_CNTLLOG}
echo 
echo " MIIDAPSAI_PATH      = ${MIIDAPSAI_PATH}" >> ${PROC_CNTLLOG}
echo " INPUTDATA_PATH      = ${INPUTDATA_PATH}" >> ${PROC_CNTLLOG}
echo " OUTPUTDATA_PATH     = ${OUTPUTDATA_PATH}" >> ${PROC_CNTLLOG}
echo 

###############################################
# main executable section 

if [ ${prep_gdas} == "True" ]; then
  echo "Preparing GDAS input files..."
  gdaslog=${LOGDIR}/${RUNID}_gdas.log
  ${PREPROCESSING_PATH}/prep_gdas.sh ${YYYYMMDD} &> ${gdaslog} &
  wait 
fi 

temp_jobid=-1
if [ ${prep_nucaps} == "True" ]; then
  echo "Preparing NUCAPS input files..."
  preplog=${LOGDIR}/${RUNID}_prep.log
  if [[ ${SUBMIT} == "SLURM" ]]; then 
    SUB_CMD="sbatch --mem=180000 --qos=${QOS} --partition=${PARTITION} --time=${WALLTIME} --job-name=prep_nucaps --output=${LOGDIR}/${RUNID}_%x.%j --ntasks=1" 
    
    temp_jobid=$(${SUB_CMD} ${PREPROCESSING_PATH}/prep_nucaps.sh ${YYYYMMDD})
    temp_jobid="$(echo ${temp_jobid} | grep -o -P 'job .{0,7}' | cut -c5-12)"
  else  
    ${PREPROCESSING_PATH}/prep_nucaps.sh ${YYYYMMDD} &> ${preplog} &
    wait 
  fi
fi

if [ ${run_inference} == "True" ]; then
  echo "Running MIIDAPS-AI inference..."
  ailog=${LOGDIR}/${RUNID}_ai.log
  if [[ ${SUBMIT} == "SLURM" ]]; then 
    SUB_CMD="sbatch --qos=${QOS} --partition=${PARTITION} --time=${WALLTIME} --job-name=run_miidaps-ai --output=${LOGDIR}/${RUNID}_%x.%j"
    if [[ ${temp_jobid} -gt 0 ]]; then
      SUB_CMD="${SUB_CMD} --dependency=afterok:${temp_jobid}"
    fi
    temp_jobid=$(${SUB_CMD} ${INFERENCERUN_PATH}/run_miidaps-ai.sh ${YYYYMMDD})
    temp_jobid="$(echo ${temp_jobid} | grep -o -P 'job .{0,7}' | cut -c5-12)"
  else
    ${INFERENCERUN_PATH}/run_miidaps-ai.sh ${YYYYMMDD} &> ${ailog} &
    wait 
  fi
fi

if [ ${postprocess_output} == "True" ]; then
  echo "Running MIIDAPS-AI postprocessing..."
  postlog=${LOGDIR}/${RUNID}_post.log
  if [[ ${SUBMIT} == "SLURM" ]]; then 
    SUB_CMD="sbatch --qos=${QOS} --partition=${PARTITION} --time=${WALLTIME} --job-name=postprocess_miidaps-ai --output=${LOGDIR}/${RUNID}_%x.%j"
    if [[ ${temp_jobid} -gt 0 ]]; then
      SUB_CMD="${SUB_CMD} --dependency=afterok:${temp_jobid} --ntasks=3"
    fi
    temp_jobid=$(${SUB_CMD} ${POSTPROCESSING_PATH}/postprocess_miidaps-ai.sh ${YYYYMMDD})
    temp_jobid="$(echo ${temp_jobid} | grep -o -P 'job .{0,7}' | cut -c5-12)"
  else
    ${POSTPROCESSING_PATH}/postprocess_miidaps-ai.sh ${YYYYMMDD} &> ${postlog} &
    wait 
  fi
fi



  



