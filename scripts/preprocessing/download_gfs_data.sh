#!/bin/sh

gDate=`date -d "${DATE}" +"%Y%m%d"`

CMDOPT=""

fLengths="anl f000 f003 f006" 
aTimes="00 06 12 18"
mResns="0p25 1p00"

oDir=${GDAS_DATA_DIR}/${gDate}
mkdir -p ${oDir}
cd ${oDir}
echo "Current Directory : ${oDir}"
for aTime in ${aTimes}; do 
  echo ${aTime}
  for fLength in ${fLengths}; do 
     echo ${fLength}
     for mRes in ${mResns}; do
       echo ${mRes}	 
       gFile=${URL}gdas.${gDate}/${aTime}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       echo ${gFile}
       bFile=gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       if [ ! -f ${bFile} ]; then 
	 ${WGETCMD} ${CMDOPT} ${gFile}
       fi
  done
  done
done

fLengths="anl f000 f003 f006" 
aTimes="18"
mResns="0p25 1p00"
gDate=`date -d "${DATE} 1 day ago" +"%Y%m%d"`

oDir=${GDAS_DATA_DIR}/${gDate}
mkdir -p ${oDir}
cd ${oDir}
for aTime in ${aTimes}; do 
  for fLength in ${fLengths}; do 
     for mRes in ${mResns}; do
       gFile=${URL}gdas.${gDate}/${aTime}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       bFile=gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       if [ ! -f ${bFile} ]; then 
         ${WGETCMD} ${CMDOPT} ${gFile}
       fi	 
  done
  done
done

fLengths="anl f000 f003 f006" 
aTimes="00"
mResns="0p25 1p00"
gDate=`date -d "${DATE} +1 day" +"%Y%m%d"`

oDir=${GDAS_DATA_DIR}/${gDate}
mkdir -p ${oDir}
cd ${oDir}
for aTime in ${aTimes}; do 
  for fLength in ${fLengths}; do 
     for mRes in ${mResns}; do
       gFile=${URL}gdas.${gDate}/${aTime}/gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       bFile=gdas.t${aTime}z.pgrb2.${mRes}.${fLength}
       if [ ! -f ${bFile} ]; then 
         ${WGETCMD} ${CMDOPT} ${gFile}
       fi
  done
  done
done

