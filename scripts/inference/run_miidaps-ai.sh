#!/bin/sh

wrtcfg(){

echo "[AI_AlgorithmParameters_Preprocessing] > ${configFile}"

echo "[AI_AlgorithmParameters_Preprocessing]" > ${configFile}
echo "PerformPCA: True" >> ${configFile}
echo "Number_Of_TB_PCs: 80, 22" >> ${configFile}
echo "PerformPCATarget: True" >> ${configFile}
echo "Number_Of_Target_PCs: 80" >> ${configFile}
echo "EmisTransform: True" >> ${configFile}

echo "[AI_AlgorithmParameters]" >> ${configFile}
echo "LossType: mean_squared_error" >> ${configFile}
echo "Optimizer: Adam" >> ${configFile}
echo "PredictionBatchSize: 100000" >> ${configFile}
echo "InstrumentSelect: 2" >> ${configFile}

echo "[AI_AlgorithmParameters_IO]" >> ${configFile}
echo "BaseDir: ${MIIDAPSAI_PATH}" >> ${configFile}
echo "DataDir: ${AIMODELDATA_PATH}" >> ${configFile}
echo "ModelDir: ${AIMODELDATA_PATH}" >> ${configFile}
echo "ModelName: ${RUNID}" >> ${configFile}
echo "PCAFile: ${PCAFiles} " >> ${configFile}
echo "InputFile: ${inputFile}" >> ${configFile}
echo "OutputDir: ${outputDir}" >> ${configFile}
echo "MinMaxFile: %(ModelDir)s/minmax.json" >> ${configFile}
echo "OutputFile: ${outputFile}" >> ${configFile}
echo "AIModelJSON: %(ModelDir)s/model.json" >> ${configFile}
echo "AIModelWeights: %(ModelDir)s/model_weights.h5" >> ${configFile}
echo "InputFileEndian: big" >> ${configFile}

}

echo "Running MIIDAPS-AI ...."
echo 

DateStart=`date`
echo "Date Start: ${DateStart}"

#---create directory for MIIDAPS-AI configuration file
export configDir=${INPUTDATA_PATH}/run/namelist/
mkdir -p ${configDir}

#---python script for MIIDAPS-AI
export pyScript=${PYTHON_RUN_PATH}/miidaps-ai_run.py

#---input file from GDAS/NUCAPS preprocessing
export inputFile=${INPUTDATA_PATH}/aifile_${YYYYMMDD}_j01_atms_j01_cris.atm

#---names of PCA json files
export PCAFiles="%(DataDir)s/pca_atms_atm.json, %(DataDir)s/pca_cris_atm.json, %(DataDir)s/pca_atms_ems.json, %(DataDir)s/pca_cris_ems.json"

#---names of configuration file and outputfile from MIIDAPS-AI
configFile=${configDir}/${RUNID}_atmems.cfg
outputFile=${RUNDIR}/${RUNID}_${YYYYMMDD}.nn

#---write configuration
wrtcfg 

#---run MIIDAPS-AI
stdbuf -oL ${PYTHON} ${pyScript} ${configFile} 

DateEnd=`date`
echo "Date End: ${DateEnd}"
