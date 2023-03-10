import numpy as np
from ai_io import read_ai
import glob 
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import re
import h5netcdf.legacyapi as hnCDF4
import sys

def read_instrconfig(instrConfigFile):
  #---read MiRS Instrument Config File
  with open(instrConfigFile, 'r') as fid:
     rLine = fid.readline()
     nChan = int(rLine[25::])
     print("nChan = ", nChan)
     nChanCnt = 0
     InstrConfig_Frequency = np.zeros((nChan), dtype=np.float32)
     for iLine, rLine in enumerate(fid):
        rLineTrim = rLine.strip() #[25::]
        #---parse line for frequencies
        fLine = [float(s) for s in re.findall(r'-?\d+\.?\d*', rLineTrim)]
        ind = np.arange(nChanCnt, nChanCnt+len(fLine))
        if (nChanCnt == nChan): break
        InstrConfig_Frequency[ind] = fLine
        nChanCnt += len(fLine)
  return nChan, InstrConfig_Frequency 

def read_channelset_file(channelSetFile):
  ind = np.loadtxt(channelSetFile)
  ChanSet_Index = (ind == 1)
  return ChanSet_Index

def get_frequency_match(NCrIS_Channels,Frequency,NInstrConfig_Channels,InstrConfig_Frequency):

  if (NInstrConfig_Channels > NCrIS_Channels): 
    print('Something wrong')
    quit()

  Index_IC = np.zeros((NInstrConfig_Channels), dtype=np.int32)
  for Channel_Index, Freq_IC in enumerate(InstrConfig_Frequency):
    ic = np.argmin(np.abs(Freq_IC - Frequency))
    Index_IC[Channel_Index] = ic 
  return Index_IC


def psig(ps):

  ak = np.array([0.000000e00, 4.804826e-02, 6.593752e00, 1.313480e01, 1.961311e01, 2.609201e01,
                 3.257081e01, 3.898201e01, 4.533901e01, 5.169611e01, 5.805321e01, 6.436264e01,
                 7.062198e01, 7.883422e01, 8.909992e01, 9.936521e01, 1.091817e02, 1.189586e02,
                 1.286959e02, 1.429100e02, 1.562600e02, 1.696090e02, 1.816190e02, 1.930970e02,
                 2.032590e02, 2.121500e02, 2.187760e02, 2.238980e02, 2.243630e02, 2.168650e02,
                 2.011920e02, 1.769300e02, 1.503930e02, 1.278370e02, 1.086630e02, 9.236572e01,
                 7.851231e01, 6.660341e01, 5.638791e01, 4.764391e01, 4.017541e01, 3.381001e01,
                 2.836781e01, 2.373041e01, 1.979160e01, 1.645710e01, 1.364340e01, 1.127690e01,
                 9.292942e00, 7.619842e00, 6.216801e00, 5.046801e00, 4.076571e00, 3.276431e00,
                 2.620211e00, 2.084970e00, 1.650790e00, 1.300510e00, 1.019440e00, 7.951341e-01,
                 6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
                 1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
                 1.000000e-02])
  bk = np.array([1.000000e00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
                 8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
                 7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
                 5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
                 2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
                 6.960025e-03, 8.175413e-09, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                 0.000000e00])
  try:
   nps = len(ps)
  except:
   nps = 1
  pres = np.tile(ak,(nps,1)) + np.tile(bk,(nps,1))*np.tile(ps,(73,1)).T
  i1 = np.arange(0,72)
  i2 = np.arange(1,73)
  psg  = 0.5*(pres[:,i1] + pres[:,i2])
  delp = np.abs(pres[:,i1] - pres[:,i2])
  return psg, delp

def read_predictions_nc(filename):
 rg = hnCDF4.Dataset(filename, "r")
 Atmosphere_Params      = rg.variables['Atmosphere_Params'][:,:]
 Cloud_Params           = rg.variables['Cloud_Params'][:,:]
 Emis_Params            = rg.variables['Emis_Params'][:,:]
 Bias_Params            = rg.variables['Bias_Params'][:,:]

 AI_Params = np.c_[Atmosphere_Params, Cloud_Params, Emis_Params, Bias_Params]
 return AI_Params

def getAIVars(aipred,watrmin,psurf):
 
 #---surface temperature
 tskinAI = aipred[:,-1]
 
 #---initial checks for negative water profiles 
 ind_watr = np.arange(72,144)
 for i in range(0,72):
   xa = aipred[:,i].squeeze()
   xa[(xa < 0)] = 0.
   aipred[:,i] = xa

 waterAI = aipred[:,ind_watr]
 for i in range(0,72):
   x = waterAI[:,i].squeeze()
   x[(x < watrmin[i])] = watrmin[i]
   waterAI[:,i] = x 

 aipred[:,ind_watr] = waterAI
 #---compute AI pressure grid
 presAI, dpAI = psig(psurf)
 presAI  = np.fliplr(presAI)
 dpAI    = np.fliplr(dpAI)
 #---compute tpw from profile amounts
 tpwAI  = np.sum(dpAI * aipred[:,ind_watr],axis=1)/10./9.8

 tempAI = aipred[:,:72]

 return tskinAI, waterAI, tempAI, tpwAI, presAI, dpAI

def interpolate_to_MiRS(psurf, presAI, presMIRSLev, presMIRSLay, tempAI, waterAI):

 nps    = len(psurf)
 print(nps, tempAI.shape)
 nmirs  = len(presMIRSLay)
 tempi  = np.zeros((nps,nmirs),dtype=np.float32) 
 wateri = np.zeros((nps,nmirs),dtype=np.float32)
 logpv  = np.log(presMIRSLev)
 logpy  = np.log(presMIRSLay)
 logpe  = np.log(presAI.squeeze())
 psmax  = np.max(presAI,axis=1)
 tiny_water = 1.e-6
 #---loop over profiles
 for ip in range(0,nps):
   #---interpolate good points (< surface pressure)
   ind          = ( (presMIRSLev <= psurf[ip]) | (presMIRSLev <= psmax[ip]) ) #& (pecm <= ps[ip]) )
   #---next try
   npri = np.argmin(np.abs(presMIRSLev - psurf[ip]))+1
   ind = np.arange(0,npri)
   water1       = np.exp(np.interp(logpv[ind],logpe[ip,:],np.log(waterAI[ip,:]))) #.squeeze())
   temp1        = np.interp(logpv[ind],logpe[ip,:],tempAI[ip,:]) #.squeeze())
   #---average profiles between pressure levels
   ind = np.arange(0,len(temp1)-1)
   tempi[ip,ind]  = 0.5*(temp1[1:] + temp1[0:-1])
   wateri[ip,ind] = 0.5*(water1[1:] + water1[0:-1])
   #---throw out below surface 
   # ind          = ( (presMIRSLay > psurf[ip]) ) #& (pecm <= ps[ip]) )
   ind          = ( (presMIRSLay > psurf[ip]) | (presMIRSLay > psmax[ip]) )
   ind = np.arange(npri-1,nmirs)
   tempi[ip,ind]  = -9999.
   wateri[ip,ind] = -9999.
   
 return tempi, wateri 

def defMiRSPressures():

  # Level pressures
  presMIRSLev = [0.005,  0.016,  0.038,  0.077,  0.137,  0.224,  0.345,  0.506,  0.714,  0.975,
                 1.297,  1.687,  2.153,  2.701,  3.340,  4.077,  4.920,  5.878,  6.957,  8.165,
                 9.512, 11.004, 12.649, 14.456, 16.432, 18.585, 20.922, 23.453, 26.183, 29.121,
                 32.274, 35.651, 39.257, 43.100, 47.188, 51.528, 56.126, 60.989, 66.125, 71.540,
                 77.240, 83.231, 89.520, 96.114,103.017,110.237,117.777,125.646,133.846,142.385,
                 151.266,160.496,170.078,180.018,190.320,200.989,212.028,223.441,235.234,247.408,
                 259.969,272.919,286.262,300.000,314.137,328.675,343.618,358.966,374.724,390.893,
                 407.474,424.470,441.882,459.712,477.961,496.630,515.720,535.232,555.167,575.525,
                 596.306,617.511,639.140,661.192,683.667,706.565,729.886,753.628,777.790,802.371,
                 827.371,852.788,878.620,904.866,931.524,958.591,986.067,1013.948,1042.232,1070.917,
                 1100.000]

  # Layer pressures :
  presMIRSLay = [0.011,  0.027,  0.057,  0.107,  0.181,  0.285,  0.425,  0.610,  0.845,  1.136,
                 1.492,  1.920,  2.427,  3.020,  3.708,  4.498,  5.399,  6.417,  7.561,  8.839,
                 10.258, 11.826, 13.552, 15.444, 17.508, 19.753, 22.188, 24.818, 27.652, 30.697,
                 33.963, 37.454, 41.178, 45.144, 49.358, 53.827, 58.557, 63.557, 68.833, 74.390,
                 80.236, 86.376, 92.817, 99.565,106.627,114.007,121.712,129.746,138.115,146.826,
                 155.881,165.287,175.048,185.169,195.655,206.508,217.734,229.337,241.321,253.689,
                 266.444,279.591,293.131,307.068,321.406,336.146,351.292,366.845,382.809,399.184,
                 415.972,433.176,450.797,468.836,487.296,506.175,525.476,545.199,565.346,585.916,
                 606.909,628.326,650.166,672.430,695.116,718.225,741.757,765.709,790.080,814.871,
                 840.079,865.704,891.743,918.195,945.057,972.329,1000.008,1028.090,1056.574,
                 1085.458]

  return np.array(presMIRSLev), np.array(presMIRSLay)

def write_nc(aiFileNetCDF,sfctypeAI,tskinAI,tempAI,waterAI,tpwAI,psurfAI,presMIRSLay,latitudeAI,longitudeAI,timeAI,qcAI):
  #---write to file 
#  rootgrp = Dataset(aiFileNetCDF,"w",format="NETCDF4")
  rootgrp = hnCDF4.Dataset(aiFileNetCDF, 'w')
  nProfile, nLayer = tempAI.shape 
  nProfile   = rootgrp.createDimension("nProfile", nProfile)
  nLayer     = rootgrp.createDimension("nLayer", nLayer)
  tprof      = rootgrp.createVariable("temperature_profile","f4",("nProfile","nLayer",))
  qprof      = rootgrp.createVariable("water_mr_profile","f4",("nProfile","nLayer",))
  tskin      = rootgrp.createVariable("skin_temperature","f4",("nProfile",))
  psurf      = rootgrp.createVariable("surface_pressure","f4",("nProfile",))
  Play       = rootgrp.createVariable("pressure_profile","f4",("nLayer",))
  alat       = rootgrp.createVariable("latitude","f4",("nProfile",))
  alon       = rootgrp.createVariable("longitude","f4",("nProfile",))
  time       = rootgrp.createVariable("time","f4",("nProfile",))
  tpw        = rootgrp.createVariable("total_precipitable_water","f4",("nProfile",))
  qc         = rootgrp.createVariable("quality_flag","u1",("nProfile",))
  sfct       = rootgrp.createVariable("surface_type","u1",("nProfile",))

  tprof[:,:]      = tempAI
  qprof[:,:]      = waterAI
  time[:]         = timeAI
  alat[:]         = latitudeAI
  alon[:]         = longitudeAI
  tpw[:]          = tpwAI
  psurf[:]        = psurfAI
  tskin[:]        = tskinAI
  Play[:]         = presMIRSLay
  qc[:]           = qcAI
  sfct[:]         = sfctypeAI
  
  rootgrp.close()
  return
  
def main(aiFileAtm,aiFilePredict,aiFileNetCDF,instrConfigFile,channelSetFile,nEOFemis,nEOFcloud):

 #---quality control limits
 tskmax = 315.

 iCnt = 0
 np.random.seed(1)
 nPrfReal, nChan, nEOF, nAnc, nEDRs, \
      npredictor, predsim, targets, tscale, logflag, sfctype, longitude, latitude, time, psurf, \
      nsensors, sensor_ids, sensor_nChan  = read_ai(aiFileAtm,input_endian='big',cnvdbl=False)


 ###############
 #---read CrIS instrument config file 
 Number_of_InstrConfig_Channels, CrIS_Frequencies = read_instrconfig(instrConfigFile)

 #---read channel subset file 
 ChanSet_Index = read_channelset_file(channelSetFile)
 CrIS_Frequencies = CrIS_Frequencies[ChanSet_Index]
 Number_of_CrIS_Channels = len(CrIS_Frequencies)

 print(aiFileAtm)
 # quit()
 predictors = predsim[:,0]
 predictors = np.min(predsim[:,0:nChan],axis=1)
 angle = predsim[:,-1]
 angle[angle < -70] = np.nan
 nEOFuAtm = 145
 nEOFu    = nEOFuAtm + nChan + nEOFemis + nEOFcloud
 #---read MIIDAPS-AI retrievals 
 aipred     = read_predictions_nc(aiFilePredict)
 nprof      = nPrfReal
 clouds     = np.zeros((nPrfReal,6),dtype=np.float32)
 nEOFuAll   = nEOFuAtm + nEOFcloud + nEOFemis
 iatm       = np.arange(0,nEOFuAtm)
 icloud     = np.arange(nEOFuAtm,nEOFuAtm+nEOFcloud)
 iemis      = np.arange(nEOFuAtm+nEOFcloud,nEOFuAll)
 ibias      = np.arange(nEOFuAll,nEOFuAll+nChan)
 aicloud    = aipred[:,icloud]
 aiemis     = aipred[:,iemis]
 bias_corp  = aipred[:,ibias]
 aipred     = aipred[:,iatm]
 watrmin    =0.95*np.array([1.08824053e-03, 1.17514282e-03, 1.32152042e-03, 1.42986944e-03, 1.56223541e-03,
               1.66054640e-03, 1.86120160e-03, 1.99159142e-03, 2.23015109e-03, 2.55793706e-03, 
               2.98136263e-03, 3.29913711e-03, 3.75220971e-03, 3.87069210e-03, 3.74018075e-03, 
               3.55569413e-03, 3.43995588e-03, 3.25411744e-03, 3.14745400e-03, 3.08026467e-03, 
               2.97272205e-03, 2.94534513e-03, 2.93187844e-03, 2.91899219e-03, 2.91142333e-03, 
               2.90899817e-03, 2.90327170e-03, 2.86783604e-03, 2.81475950e-03, 2.68201390e-03, 
               2.43698899e-03, 2.28727260e-03, 2.17958307e-03, 2.10954854e-03, 2.04813248e-03, 
               1.72303116e-03, 9.24269145e-04, 7.35406938e-04, 1.01662509e-03, 1.54576648e-03, 
               1.27134193e-03, 1.02637731e-03, 1.92272675e-03, 4.64830489e-04, 3.64751089e-03, 
               3.30091920e-03, 4.38262988e-03, 1.74221164e-03, 3.74655100e-03, 3.11210752e-03, 
               3.44836758e-03, 6.37486856e-03, 1.55607555e-02, 1.00481082e-02, 4.94700391e-03, 
               5.30258752e-03, 2.61755267e-05, 1.25908424e-04, 2.95943901e-04, 1.39814503e-02, 
               3.86493206e-02, 5.61893135e-02, 4.75064442e-02, 4.16557863e-02, 5.60530648e-02, 
               6.73312247e-02, 6.15505539e-02, 5.47851659e-02, 4.79164831e-02, 4.05063890e-02, 
               2.60247160e-02, 0.00000000e+00])

 #---parse out variables and perform initial QC on atmospheric retrievals profiles
 tskinAI, waterAI, tempAI, tpwAI, presAI, dpAI = getAIVars(aipred,watrmin,psurf)
 iall     = ( tskinAI < tskmax ) & ( tskinAI > 0 ) & \
            ( predictors > 0 ) & ( (sfctype == 0 ) | (sfctype == 1) | (sfctype == 2) | (sfctype == 3) )

 presAI   = presAI[iall,:]; dpAI = dpAI[iall,:]
 psurfAI  = psurf[iall]
 sfctype  = sfctype[iall]

 tpwAI    = tpwAI[iall] 
 tskinAI  = tskinAI[iall] 
 tempAI   = tempAI[iall,:]; waterAI = waterAI[iall,:]

 latitude = latitude[iall]; longitude = longitude[iall]
 time     = time[iall]
 angle    = angle[iall]
 
 #---get MIRS pressure levels and layers
 presMIRSLev, presMIRSLay = defMiRSPressures()
 #---interpolate/average profiles onto MiRS Layers
 tempMIRS, waterMIRS = interpolate_to_MiRS(psurfAI, presAI, presMIRSLev, presMIRSLay, tempAI, waterAI)

 #---write combined data to file 
 nps    = len(psurfAI)
 qc = np.zeros((nps),dtype=np.uint8)
 print("len=", len(psurf), len(qc))
 sfctype = sfctype.astype(np.uint8)
 write_nc(aiFileNetCDF,sfctype,tskinAI,tempMIRS,waterMIRS,tpwAI,psurfAI,presMIRSLay,latitude,longitude,time,qc)

nEOFemis      = 22+366
nEOFcloud     = 4
aiFileAtm     = sys.argv[1]
aiFilePredict = sys.argv[2]
aiFileNetCDF  = sys.argv[3]
instrConfigFile = sys.argv[4]
channelSetFile = sys.argv[5]

main(aiFileAtm,aiFilePredict,aiFileNetCDF,instrConfigFile,channelSetFile,nEOFemis,nEOFcloud)


