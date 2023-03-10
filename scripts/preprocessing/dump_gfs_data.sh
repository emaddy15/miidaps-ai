#!/bin/sh

gDate=`date -d "${DATE}" +"%Y%m%d"`

fLengths="f000" 
aTimes="00 06 12 18"
mResns="0p25 1p00"

oDir=${GDAS_DATA_DIR}/${gDate}
dDir=${GDAS_DUMP_DIR}/${gDate}
logDir=${dDir}/log/
mkdir -p ${dDir}
mkdir -p ${logDir}
cd ${dDir}

for aTime in ${aTimes}; do 
  for fLength in ${fLengths}; do 
     for mRes in ${mResns}; do
       dumpOut=${logDir}dump_${gDate}.t${aTime}z.${mRes}.${fLength}.log
       gFile=${oDir}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       oFile=${dDir}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}.bin
       if [ -s ${gFile} ]; then
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":PRES:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":ICEC:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":SNOD:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":LAND:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":TMP:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         echo "${WGRIB2_CMD} ${gFile} 2> ${dumpOut} | grep ":PRES:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -bin ${oFile}"
       fi
  done
 done
done 

gDate=`date -d "${DATE} 1 day ago" +"%Y%m%d"`
fLengths="f000" 
aTimes="18"
mResns="0p25 1p00"

oDir=${GDAS_DATA_DIR}/${gDate}
dDir=${GDAS_DUMP_DIR}/${gDate}
logDir=${dDir}/log/
mkdir -p ${dDir}
mkdir -p ${logDir}
cd ${dDir}
for aTime in ${aTimes}; do 
  for fLength in ${fLengths}; do 
     for mRes in ${mResns}; do
       dumpOut=${logDir}dump_${gDate}.t${aTime}z.${mRes}.${fLength}.log
       gFile=${oDir}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       oFile=${dDir}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}.bin
       if [ -s ${gFile} ]; then
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":PRES:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":ICEC:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":SNOD:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":LAND:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":TMP:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         echo "${WGRIB2_CMD} ${gFile} 2> ${dumpOut} | grep ":PRES:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -bin ${oFile}"
       fi
  done
 done
done 

fLengths="anl f000 f003 f006" 
aTimes="00"
mResns="0p25 1p00"
gDate=`date -d "${DATE} +1 day" +"%Y%m%d"`
oDir=${GDAS_DATA_DIR}/${gDate}
dDir=${GDAS_DUMP_DIR}/${gDate}
logDir=${dDir}/log/
mkdir -p ${dDir}
mkdir -p ${logDir}
cd ${dDir}
for aTime in ${aTimes}; do 
  for fLength in ${fLengths}; do 
     for mRes in ${mResns}; do
       dumpOut=${logDir}dump_${gDate}.t${aTime}z.${mRes}.${fLength}.log
       gFile=${oDir}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       oFile=${dDir}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}.bin
       if [ -s ${gFile} ]; then
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":PRES:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":ICEC:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":SNOD:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":LAND:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         ${WGRIB2_CMD} -s ${gFile} 2> ${dumpOut} | grep ":TMP:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -append -bin ${oFile}
         echo "${WGRIB2_CMD} ${gFile} 2> ${dumpOut} | grep ":PRES:surface:" |  ${WGRIB2_CMD} -i ${gFile} -no_header -order we:ns -bin ${oFile}"
       fi
  done
 done
done 
