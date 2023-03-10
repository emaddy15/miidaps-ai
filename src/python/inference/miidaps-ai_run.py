from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import urllib
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import json 
#import ConfigParser 
from configparser import ConfigParser 
import os
import sys
import os.path 

import keras 
from keras.models import model_from_json
from keras.utils import to_categorical
from keras import backend as K
from keras import optimizers

from keras.models import Sequential, Model, Input
from keras.layers import Flatten
from keras.layers import Dense, Reshape
from keras.layers import Dropout

from ai_io import read_ai
import time 

ONE_FP=1.
ZERO_FP=0.
OCEAN_TYPE=0
TB_Min_Value = 50.
TB_Max_Value = 550.
Angle_Max_Norm = 65.
Index_Water_Start = 72
Index_Water_End   = 144
Default_Bad_Value = np.nan
Default_Real_Type = np.float32
Default_Random_Seed = 42
Number_Of_Cloud_EOFs = 4
mainOutputLossWeight = ONE_FP
biasOutputLossWeight = 100.
cloudOutputLossWeight = 500.0
AI_Predictor_Threshold = -100.
Number_Of_ATMS_Channels = 22 
Default_Real_BigEndian_Type = np.dtype('>f4')

def rescale_prescale(a,b,mnx,mxx,x):
 """Rescale variable between a, b using predefined 
 max and min (mxx,mnx)
 Arguments: 
 a -- output variable maximum 
 b -- output variable minimum 
 mnx -- predefined minimum of x
 mxx -- predefined maximum of x 
 """ 
 xp = a + ( b - a ) * ( x - mnx ) / ( mxx - mnx ) 
 return xp

def transform_predictors(predictors,nsensors,sensor_nchan,pca_1,pca_2,npc_1,npc_2):
 nall = sensor_nchan[0]+sensor_nchan[1]
 i1  = np.arange(0,sensor_nchan[0])
 i2  = np.arange(sensor_nchan[0],sensor_nchan[0]+sensor_nchan[1])
 ps  = predictors[:,-1]
 p1  = predictors[:,i1]
 p2  = predictors[:,i2]
 p1  = np.c_[p1, ps]
 p2  = np.c_[p2, ps]
 p1t = pca_1.transform(p1)
 p2t = pca_2.transform(p2)
 i1t = np.arange(0,npc_1)
 i2t = np.arange(npc_1,npc_1+npc_2)
 nprf,nx = predictors.shape
 predictorst = np.zeros((nprf,npc_1+npc_2),dtype=np.float32)
 predictorst[:,i1t] = p1t
 predictorst[:,i2t] = p2t
 return predictorst

def get_config_defaults():
 defaults = {
  'PerformPCA': 'true',
  'Number_Of_TB_PCs': '80, 22', 
  'PerformPCATarget': 'true',
  'Number_Of_Target_PCs': '80',
  'Standardize_TQ': 'false',
  'Standardize_Emis': 'false',
  'LossType': 'mean_squared_error',
  'Optimizer': 'Adam',
  'PredictionBatchSize': '100000',
  'BaseDir': '',
  'DataDir': '',
  'ModelDir': '%(BaseDir)s/%(DataDir)s/',
  'ModelName': 'j01_crisatms_v001',
  'PCAFiles': '%(DataDir)s/pca_atms_atm.json, %(DataDir)s/pca_cris_atm.json, %(DataDir)s/pca_atms_ems.json, %(DataDir)s/pca_cris_ems.json',
  'InputFile': 'test_crisatms.atm',
  'OutputDir': '',
  'MinMaxFile': '%(ModelDir)s/minmax.json',
  'OutputFile': '%(ModelDir)s/%(ModelName)s_combine.nn',
  'AIModelJSON': '%(ModelDir)s/model.json',
  'AIModelWeights': '%(ModelDir)s/model_weights.h5',
  'InputFileEndian': 'big',
  'InstrumentSelect': '2',
 }
 return defaults

def normalizePredictors(Number_Of_Profiles, mnt_CrIS, mxt_CrIS, AIPredictors_CrIS, mnt_ATMS, mxt_ATMS, AIPredictors_ATMS):
 #---restrict range to 0., 1
 for i in range(0,Number_Of_Profiles):
   x  = AIPredictors_CrIS[i,:].squeeze()
   xp = rescale_prescale(-ONE_FP,ONE_FP,mnt_CrIS,mxt_CrIS,x)
   AIPredictors_CrIS[i,:] = xp 

   x  = AIPredictors_ATMS[i,:].squeeze()
   xp = rescale_prescale(-ONE_FP,ONE_FP,mnt_ATMS,mxt_ATMS,x)
   AIPredictors_ATMS[i,:] = xp 

 return AIPredictors_CrIS, AIPredictors_ATMS

def inverse_transform_predictions(pcat_components_temp,pcat_mean_temp,pcat_components_watr,pcat_mean_watr,
                                  index_temp,index_watr,watr_pca_scale,predictions):

  print('pc_temp=',pcat_components_temp.shape)
  print('pc_temp=',pcat_mean_temp.shape)
  print('pc_watr=',pcat_components_watr.shape)
  print('pc_watr=',pcat_mean_watr.shape)
  print('watr_s=',watr_pca_scale.shape)
  print('indx_t=',index_temp.shape)
  print('indx_q=',index_watr.shape)
  predictions_temp = pca_inverse_transform(pcat_components_temp,pcat_mean_temp,predictions[:,index_temp])
  predictions_watr = pca_inverse_transform(pcat_components_watr,pcat_mean_watr,predictions[:,index_watr]/watr_pca_scale)

  return predictions_temp, predictions_watr

def load_atm_json(jsonFile):
 with open(jsonFile,'r') as fid:
  atm_data = json.load(fid)

 Number_Of_TB_PCs = atm_data['Number_Of_TB_PCs'] 
 Number_Of_Target_Temp_PCs = atm_data['Number_Of_Target_Temp_PCs'] 
 Number_Of_Target_Watr_PCs = atm_data['Number_Of_Target_Watr_PCs'] 
 water_scale = np.array(atm_data['water_scale'])
 pca_components = np.array(atm_data['pca_components'])
 pca_mean       = np.array(atm_data['pca_mean'])
 pcat_Temp_components = np.array(atm_data['pca_Temp_components'] )
 pcat_Temp_mean       = np.array(atm_data['pca_Temp_mean'] )
 pcat_Watr_components = np.array(atm_data['pca_Watr_components'])
 pcat_Watr_mean       = np.array(atm_data['pca_Watr_mean'])
 mnt = np.array(atm_data['mnt'] )
 mxt = np.array(atm_data['mxt'] )
 index_temp = atm_data['index_temp']
 index_watr = atm_data['index_watr']
 return Number_Of_TB_PCs, Number_Of_Target_Temp_PCs, Number_Of_Target_Watr_PCs, water_scale, pca_components, pca_mean, \
  pcat_Temp_components, pcat_Temp_mean, pcat_Watr_components, pcat_Watr_mean, mnt, mxt, index_temp, index_watr

def load_emis_json(jsonFile):
 with open(jsonFile,'r') as fid:
  emis_data = json.load(fid)

 Number_Of_Emis_TB_PCs = emis_data['Number_Of_Emis_TB_PCs'] 
 Number_Of_Target_Emis_PCs = emis_data['Number_Of_Target_Emis_PCs'] 
 pca_Emis_components = np.array(emis_data['pca_Emis_components'] )
 pca_Emis_mean = np.array(emis_data['pca_Emis_mean']) 
 mnt = np.array(emis_data['mnt']) 
 mxt = np.array(emis_data['mxt']) 
 return Number_Of_Emis_TB_PCs, Number_Of_Target_Emis_PCs, pca_Emis_components, pca_Emis_mean, mnt, mxt

def load_minmax_json(jsonFile):
 with open(jsonFile,'r') as fid:
  minmax_data = json.load(fid)
 Number_Of_Channels_NX = minmax_data['Number_Of_Channels_NX'] 
 Number_Of_Cloud_EOFs = minmax_data['Number_Of_Cloud_EOFs'] 
 bias_minmax  = np.array(minmax_data['bias_minmax'])
 cloud_minmax = np.array(minmax_data['cloud_minmax'])
 watr_pca_scale = np.array(minmax_data['watr_pca_scale'])

 return Number_Of_Channels_NX, Number_Of_Cloud_EOFs, bias_minmax, cloud_minmax, watr_pca_scale
 
def pca_transform_data(pca_components,pca_mean,X):
  #X = np.asarray(X,dtype=np.float64)
  X = X - pca_mean
  X_transformed = np.dot(X, pca_components.T)
  return X_transformed

def pca_inverse_transform(pca_components,pca_mean,X):
  #X = np.asarray(X,dtype=np.float64)
  X_transformed = np.dot(X, pca_components)
  X_transformed = X_transformed + pca_mean
  return X_transformed

def write_nc(outputFile, Predictions, Predictions_bias, Predictions_cloud, Predictions_emis):
 
  #  pb = Predictions_bias[i,:].squeeze()
  #  pc = Predictions_cloud[i,:].squeeze()
  #  pe = Predictions_emis[i,:].squeeze()
  Number_of_Observations, Number_of_Atmospheric_Params = Predictions.shape 
  Number_of_Observations, Number_of_Bias_Params = Predictions_bias.shape 
  Number_of_Observations, Number_of_Cloud_Params = Predictions_cloud.shape 
  Number_of_Observations, Number_of_Emis_Params = Predictions_emis.shape 
  import h5netcdf.legacyapi as hnCDF4
  rootgrp   = hnCDF4.Dataset(outputFile,"w")
  n_obs     = rootgrp.createDimension("Number_of_Observations", Number_of_Observations)
  n_atm     = rootgrp.createDimension("Number_of_Atmospheric_Params", Number_of_Atmospheric_Params)
  n_ems     = rootgrp.createDimension("Number_of_Emis_Params", Number_of_Emis_Params)
  n_bis     = rootgrp.createDimension("Number_of_Bias_Params", Number_of_Bias_Params)
  n_cld     = rootgrp.createDimension("Number_of_Cloud_Params", Number_of_Cloud_Params)
  atm_p     = rootgrp.createVariable("Atmosphere_Params","f4",("Number_of_Observations","Number_of_Atmospheric_Params",))
  ems_p     = rootgrp.createVariable("Emis_Params","f4",("Number_of_Observations","Number_of_Emis_Params",))
  cld_p     = rootgrp.createVariable("Cloud_Params","f4",("Number_of_Observations","Number_of_Cloud_Params",))
  bis_p     = rootgrp.createVariable("Bias_Params","f4",("Number_of_Observations","Number_of_Bias_Params",))

  atm_p[:,:] = Predictions
  ems_p[:,:] = Predictions_emis
  cld_p[:,:] = Predictions_cloud
  bis_p[:,:] = Predictions_bias

  rootgrp.close()
  return


def main(configFile):
 """Run AI algorithm.

 Arguments:
 configFile -- name of AI configuration file
 """
 
 #---open AI configuration JSON file
 #with open(configFile,'r') as fid: 
 #  configParams = json.load(fid)

 #---open AI configuration using configparser module
 # config = ConfigParser.ConfigParser()
 configDefaults = get_config_defaults()

 config = ConfigParser(configDefaults)
 config.read(configFile)

 #---AI input preprocessing flags and variables
 PerformPCAs = config['AI_AlgorithmParameters_Preprocessing']['PerformPCA'].lower()

 standardize_TQ = config['AI_AlgorithmParameters_Preprocessing']['Standardize_TQ'].lower()
 Standardize_TQ = False
 if standardize_TQ == 'true':
  Standardize_TQ=True

 standardize_Emis = config['AI_AlgorithmParameters_Preprocessing']['Standardize_Emis'].lower()
 Standardize_Emis = False
 if standardize_Emis == 'true':
  Standardize_Emis=True

 PerformPCA=False
 if PerformPCAs == 'true':
   PerformPCA=True
 Number_Of_TB_PCs = sum(map(int,config['AI_AlgorithmParameters_Preprocessing']['Number_Of_TB_PCs'].split(',')))
 PerformPCATargets = config['AI_AlgorithmParameters_Preprocessing']['PerformPCATarget'].lower()

 PerformPCATarget=False
 if PerformPCATargets == 'true':
   PerformPCATarget=True
 print("PCA= ",PerformPCATarget)
 Number_Of_Target_PCs = int(config['AI_AlgorithmParameters_Preprocessing']['Number_Of_Target_PCs'])

 emistransform = False
 EmisTransform = config['AI_AlgorithmParameters_Preprocessing']['EmisTransform'].lower()
 if EmisTransform == 'true':
   emistransform=True

 #---model parameters
 LossType = config['AI_AlgorithmParameters']['LossType'] 
 Optimizer = config['AI_AlgorithmParameters']['Optimizer'] 
 PredictionBatchSize = int(config['AI_AlgorithmParameters']['PredictionBatchSize'] )

 #---instrument selection tuning parameters
 #   0: ATMS
 #   1: CrIS
 #   2: ATMS + CrIS
 # --- feature is deprecated in this latest incarnation 
 #     all retrievals are produced simultaneously
 InstrumentSelect = int(config['AI_AlgorithmParameters']['InstrumentSelect'])
 
 #---IO parameters
 PCAFiles = [y.strip() for y in config['AI_AlgorithmParameters_IO']['PCAFile'].split(',')]
 #PCAFiles = [y.strip for y in PCAFiles]
 InputFile = config['AI_AlgorithmParameters_IO']['InputFile'] 
 OutputDir = config['AI_AlgorithmParameters_IO']['OutputDir'] 
 ModelName = config['AI_AlgorithmParameters_IO']['ModelName'] 
 MinMaxFile = config['AI_AlgorithmParameters_IO']['MinMaxFile'] 
 OutputFile = config['AI_AlgorithmParameters_IO']['OutputFile'] 
 AIModelJSON = config['AI_AlgorithmParameters_IO']['AIModelJSON'] 
 AIModelWeights = config['AI_AlgorithmParameters_IO']['AIModelWeights'] 
 InputFileEndian = config['AI_AlgorithmParameters_IO']['InputFileEndian']
 print('config=',config)
 print('AIModelWeights=',AIModelWeights)
 print('AIModelJSON=',AIModelJSON)
 print('AIInputFile=',InputFile)
 print('PCAFiles   =',PCAFiles)
 print('MinMaxFile =',MinMaxFile)
 Number_Of_Observations_Skip = 1

 select_cris = False
 select_atms = False
 if InstrumentSelect == 0: 
  Number_Of_Observations_Skip = 9
  select_cris = False
  select_atms = True
 elif InstrumentSelect == 1: 
  select_cris = True
  select_atms = False
 elif InstrumentSelect == 2: 
  select_cris = True
  select_atms = True
 else:
  quit()

 tic = time.time()
 #---read input file
 Number_Of_Profiles, Number_Of_Channels, Number_Of_EOFs, Number_Of_Ancillary_Params, Number_Of_EDRs, \
   Number_Of_Predictors, AIPredictors, AITargets, TargetScaling, TargetLogarithmFlag, Surface_Type, \
   Longitude, Latitude, Time, Surface_Pressure, \
   Number_Of_Sensors, Sensor_IDs, Number_Of_Sensor_Channels = read_ai(InputFile,input_endian=InputFileEndian,nskip=Number_Of_Observations_Skip)
 toc = time.time()
 print('MIIDAPS-AI RUN: Read time = ', (toc - tic)/60)
 print('Number of Sensor Channels = ', Number_Of_Sensor_Channels)
 Number_Of_Sensor_Channels[0] = Number_Of_ATMS_Channels
 Number_Of_Sensor_Channels[1] = Number_Of_Channels - Number_Of_Sensor_Channels[0]
 print('Number of Sensor Channels = ', Number_Of_Sensor_Channels)
 Number_Of_Emissivity_EOFs = Number_Of_Channels

 #---read min/max parameters from file
 Number_Of_Channels_NX, Number_Of_Cloud_EOFs, bias_minmax, cloud_minmax, watr_pca_scale = load_minmax_json(MinMaxFile)
 #---normalize ancillary predictors - use simple fixed scaling so it doesn't need to be saved on output
 angles = AIPredictors[:,-2::]
 anglers = angles / Angle_Max_Norm
 Surface_Type_Encoded = to_categorical(Surface_Type,num_classes=4)
 AIPredictors_Ancillary = np.c_[Surface_Type_Encoded, anglers]
 
 #---remove lattitude from predictor
 AIPredictors = np.c_[AIPredictors[:,:Number_Of_Channels], AIPredictors[:,Number_Of_Channels+1::]]
 Number_Of_Ancillary_Params = Number_Of_Ancillary_Params - 1 

 #---remove angle from predictor
 AIPredictors = AIPredictors[:,:-2]
 Number_Of_Ancillary_Params = Number_Of_Ancillary_Params - 2 

 print("InstrumentSelect=", InstrumentSelect)
 #---Instrument Selection resizing
 Index_Sat_Chan = np.arange(0,Number_Of_Sensor_Channels[0])
 AIPredictors_ATMS = np.c_[AIPredictors[:,Index_Sat_Chan], AIPredictors[:,-1]]
 Index_Sat_Chan = Number_Of_Sensor_Channels[0] + np.arange(0,Number_Of_Sensor_Channels[1])
 AIPredictors_CrIS = np.c_[AIPredictors[:,Index_Sat_Chan], AIPredictors[:,-1]]
 
 Number_Of_TB_PCs = np.min([Number_Of_TB_PCs, Number_Of_Channels+Number_Of_Ancillary_Params])
 Number_Of_Profiles_sav = Number_Of_Profiles  

 #---extra check for negative TB/radiances in any channel
 MinPredictor = np.min(AIPredictors[:,:Number_Of_Channels],axis=1) 
 MinTargets_atm = np.min(AITargets,axis=1)
 print("MinPredictor = ", MinPredictor)
 print("AITargets[:,0] = ", AITargets[:,0])
 
 print("AIPredictors.Shape=",AIPredictors.shape)
 Index_Good_Initial_QC = np.arange(Number_Of_Profiles)
 Index_Good_Initial_QC = (( AIPredictors[:,0] > TB_Min_Value ) & ( AIPredictors[:,0] < TB_Max_Value) & \
                          ( MinPredictor > TB_Min_Value ) & ( AIPredictors[:,-1] > AI_Predictor_Threshold) & (AITargets[:,0] > 0) & \
                          ( MinTargets_atm > ZERO_FP ))
 #Index_Good_Initial_QC = Index_Good_Initial_QC[::4] 
 #---set non-ocean biases to zero
 Index_Land = (( AIPredictors[:,0] > TB_Min_Value ) & ( MinPredictor > TB_Min_Value ) & ( AIPredictors[:,0] < TB_Max_Value) & \
                ( AIPredictors[:,-1] > AI_Predictor_Threshold) & (AITargets[:,0] > 0) & (Surface_Type != OCEAN_TYPE) )
 Ocean_TB_Mask = np.zeros((Number_Of_Profiles,Number_Of_Channels),dtype=Default_Real_Type) + 1
 Ocean_TB_Mask[Index_Land,:] = ZERO_FP
 
 Number_Of_Good_QC_Profiles = Index_Good_Initial_QC.sum()
 print('Number_Of_Good_QC_Profiles =', Number_Of_Good_QC_Profiles, Number_Of_Profiles)

 if (Number_Of_Good_QC_Profiles < 1000): 
   print('Warning total number of training + test + validation < 1000')
   quit()

 Number_Of_Profiles = Index_Good_Initial_QC.sum(0)
 Time = Time[Index_Good_Initial_QC]
 Latitude = Latitude[Index_Good_Initial_QC]
 Longitude = Longitude[Index_Good_Initial_QC]
 AITargets = AITargets[Index_Good_Initial_QC,:]
 AIPredictors = AIPredictors[Index_Good_Initial_QC,:]
 AIPredictors_CrIS = AIPredictors_CrIS[Index_Good_Initial_QC,:]
 AIPredictors_ATMS = AIPredictors_ATMS[Index_Good_Initial_QC,:]
 Surface_Type = Surface_Type[Index_Good_Initial_QC]
 Ocean_TB_Mask = Ocean_TB_Mask[Index_Good_Initial_QC,:]
 Surface_Pressure = Surface_Pressure[Index_Good_Initial_QC]
 AIPredictors_Ancillary = AIPredictors_Ancillary[Index_Good_Initial_QC,:]

 Index_Latitude = Number_Of_Channels
 Latitude = AIPredictors[:,Index_Latitude].squeeze()

 # print('Latitude - ', np.min(Latitude), np.max(Latitude))

 # for ip in range(0,Number_Of_Channels):
 #   print('ip=%d, ppmx=%f, %f'%(ip,np.min(AIPredictors[:,ip]),np.max(AIPredictors[:,ip])))
 # for ip in range(Number_Of_Channels,Number_Of_Channels+Number_Of_Ancillary_Params):
 #   print('ip=%d, ppmx=%f, %f'%(ip,np.min(AIPredictors[:,ip]),np.max(AIPredictors[:,ip])))
 
 atm_ATMS = load_atm_json(PCAFiles[0])
 Number_Of_TB_PCs_ATMS, Number_Of_Target_Temp_PCs, Number_Of_Target_Watr_PCs, water_scale_ATMS, pca_components_ATMS, pca_mean_ATMS, pcat_Temp_components_ATMS, pcat_Temp_mean_ATMS, pcat_Watr_components_ATMS, pcat_Watr_mean_ATMS, mnt_ATMS, mxt_ATMS, index_ATMS_temp, index_ATMS_watr = atm_ATMS

 atm_CrIS = load_atm_json(PCAFiles[1])
 Number_Of_TB_PCs_CrIS, Number_Of_Target_Temp_PCs, Number_Of_Target_Watr_PCs, water_scale_CrIS, pca_components_CrIS, pca_mean_CrIS, pcat_Temp_components_CrIS, pcat_Temp_mean_CrIS, pcat_Watr_components_CrIS, pcat_Watr_mean_CrIS, mnt_CrIS, mxt_CrIS, index_CrIS_temp, index_CrIS_watr = atm_CrIS

 emis_ATMS = load_emis_json(PCAFiles[2])
 Number_Of_Emis_TB_PCs_ATMS, Number_Of_Target_Emis_PCs_ATMS, pca_Emis_components_ATMS, pca_Emis_mean_ATMS, mnt_Emis_ATMS, mxt_Emis_ATMS = emis_ATMS  

 emis_CrIS = load_emis_json(PCAFiles[3])
 Number_Of_Emis_TB_PCs_CrIS, Number_Of_Target_Emis_PCs_CrIS, pca_Emis_components_CrIS, pca_Emis_mean_CrIS, mnt_Emis_CrIS, mxt_Emis_CrIS = emis_CrIS  

 #---indices of temperature and water pcs in the predictor arrays
 index_temp_pca = np.arange(0,Number_Of_Target_Temp_PCs)
 index_watr_pca = np.arange(Number_Of_Target_Temp_PCs,Number_Of_Target_Temp_PCs+Number_Of_Target_Watr_PCs)

 #---index of ATMS and CrIS emissivity pcs in the predictor arrays
 index_emis_ATMS_pca = np.arange(0,Number_Of_Target_Emis_PCs_ATMS)
 index_emis_CrIS_pca = np.arange(Number_Of_Target_Emis_PCs_ATMS,Number_Of_Target_Emis_PCs_ATMS+Number_Of_Target_Emis_PCs_CrIS)

 tic_preprocess = time.time()

 #---target water scaling 
 Index_Water = np.arange(Index_Water_Start,Index_Water_End)
 AITargets[:,Index_Water] = np.log(AITargets[:,Index_Water])
 AITargets[(~np.isfinite(AITargets))] = 1.e-10
 print('NUMBER_OF_CHANNELS = ', Number_Of_Channels)
 if PerformPCATarget == True:

  targets_temp = pca_transform_data(pcat_Temp_components_ATMS,pcat_Temp_mean_ATMS,AITargets[:,index_ATMS_temp])
  targets_watr = watr_pca_scale*pca_transform_data(pcat_Watr_components_ATMS,pcat_Watr_mean_ATMS,AITargets[:,index_ATMS_watr])
  AITargets = np.c_[targets_temp,targets_watr]

 if PerformPCA == True:
  print('...Performing PCA on predictors')
  #---do we want to add another random number to predictor tbs? 
  #---set seed this Time user defined
  np.random.seed(Default_Random_Seed)
  print(Number_Of_Profiles, len(AIPredictors), AIPredictors.shape)

  if Standardize_TQ:
    AIPredictors_CrIS = scaler_CrIS.transform(AIPredictors_CrIS)
    AIPredictors_ATMS = scaler_ATMS.transform(AIPredictors_ATMS)
  #---transform predictors for ATMS and CrIS
  AIPredictors_ATMS = pca_transform_data(pca_components_ATMS,pca_mean_ATMS,AIPredictors_ATMS)
  AIPredictors_CrIS = pca_transform_data(pca_components_CrIS,pca_mean_CrIS,AIPredictors_CrIS)

  print('...Finished performing PCA on predictors')

 #Predictors_CrIS = AIPredictors_CrIS.copy()
 #Predictors_ATMS = AIPredictors_ATMS.copy()
 #---restrict range of predictors to -1,1
# print("AI_CrIS min= ", np.nanmin(AIPredictors_CrIS,axis=0))
# print("AI_CrIS max= ", np.nanmax(AIPredictors_CrIS,axis=0))
# print("AI_CrIS 0  = ", AIPredictors_CrIS[0,:])
# print("AI_CrIS 1e4= ", AIPredictors_CrIS[10000,:])
# print("AI_ATMS min= ", np.nanmin(AIPredictors_ATMS,axis=0))
# print("AI_ATMS max= ", np.nanmax(AIPredictors_ATMS,axis=0))
# print("AI_ATMS 0  = ", AIPredictors_ATMS[0,:])
# print("AI_ATMS 1e4= ", AIPredictors_ATMS[10000,:])

 AIPredictors_CrIS, AIPredictors_ATMS = normalizePredictors(Number_Of_Profiles, 
                                                            mnt_CrIS, mxt_CrIS, AIPredictors_CrIS, mnt_ATMS, mxt_ATMS, AIPredictors_ATMS)

 toc = time.time()
 print('MIIDAPS-AI RUN: Prep time = ', (toc - tic_preprocess)/60)
 print("AI_CrIS min= ", np.nanmin(AIPredictors_CrIS,axis=0))
 print("AI_CrIS max= ", np.nanmax(AIPredictors_CrIS,axis=0))
 print("AI_CrIS 0  = ", AIPredictors_CrIS[0,:])
 print("AI_CrIS 1e4= ", AIPredictors_CrIS[10000,:])
 print("AI_ATMS min= ", np.nanmin(AIPredictors_ATMS,axis=0))
 print("AI_ATMS max= ", np.nanmax(AIPredictors_ATMS,axis=0))
 print("AI_ATMS 0  = ", AIPredictors_ATMS[0,:])
 print("AI_ATMS 1e4= ", AIPredictors_ATMS[10000,:])

 #---load json and create model
 with open(AIModelJSON, 'r') as json_file:
   loaded_model_json = json_file.read()

 model = model_from_json(loaded_model_json)
 #---load model weights
 model.load_weights(AIModelWeights)

 losses = {'main_output_atms': LossType, \
           'main_output_cris': LossType, \
           'bias_output': LossType, \
           'main_output': LossType, 'cloud_output': LossType, 'emis_output': LossType}
 
 lossWeights = {'main_output_atms': 1.0, \
                'main_output_cris': 1.0,\
                'bias_output': 100.0, \
                'main_output': 1.0, 'cloud_output': 1000.0, 'emis_output': 100.0}

 #---compile model
 model.compile(optimizer=Optimizer, loss=LossType, loss_weights=lossWeights, metrics=[LossType]) 
 print('Loaded model from disk: ', AIModelWeights, AIModelJSON)

 #---predict EDRs from input and ancillary data
 (Predictions_ATMS, Predictions_CrIS, Predictions_bias, Predictions, Predictions_cloud, Predictions_emis) = model.predict([AIPredictors_ATMS,AIPredictors_CrIS,AIPredictors_Ancillary],batch_size=PredictionBatchSize)

 #print('Predictors = ',Predictors)
 #print('Number_Of_Profiles = %d, Number_Of_Channels=%d, Number_Of_EDRs=%d, Number_Of_Predictors=%d, ntargets=%d'%(Number_Of_Profiles,Number_Of_Channels,Number_Of_EDRs,Number_Of_Predictors,Number_Of_EOFs))
 print(model.evaluate([AIPredictors_ATMS,AIPredictors_CrIS,AIPredictors_Ancillary],[AITargets,AITargets,Predictions_bias,AITargets,Predictions_cloud,Predictions_emis]))
 #---summary of model architecture
 print(model.summary())
 print('sizes=',Predictions_bias.shape,Predictions_emis.shape,Predictions_cloud.shape,Predictions.shape,Index_Good_Initial_QC.shape)
 
# print('Predictions min=',np.nanmin(Predictions,axis=0))
# print('Predictions max=',np.nanmax(Predictions,axis=0))
# print('Predictions p50=',np.nanpercentile(Predictions,[50.],axis=0))

# print('PredictionsCmin=',np.nanmin(Predictions_CrIS,axis=0))
# print('PredictionsCmax=',np.nanmax(Predictions_CrIS,axis=0))
# print('PredictionsCp50=',np.nanpercentile(Predictions_CrIS,[50.],axis=0))

# print('PredictionsAmin=',np.nanmin(Predictions_ATMS,axis=0))
# print('PredictionsAmax=',np.nanmax(Predictions_ATMS,axis=0))
# print('PredictionsAp50=',np.nanpercentile(Predictions_ATMS,[50.],axis=0))

# print('PredictionsEmin=',np.nanmin(Predictions_emis,axis=0))
# print('PredictionsEmax=',np.nanmax(Predictions_emis,axis=0))
# print('PredictionsEp50=',np.nanpercentile(Predictions_emis,[50.],axis=0))

 #---convert pca Predictions to Predictions and 
 if PerformPCATarget == True:
   print("performing inverse transform")
   predictions_temp, predictions_watr = inverse_transform_predictions(pcat_Temp_components_ATMS,pcat_Temp_mean_ATMS,\
                                                                      pcat_Watr_components_ATMS,pcat_Watr_mean_ATMS,\
                                                                      index_temp_pca,index_watr_pca,watr_pca_scale,Predictions)
   predictions_watr = np.exp(predictions_watr)
   Predictions = np.c_[predictions_temp[:,:72],predictions_watr,predictions_temp[:,-1]]

   predictions_temp, predictions_watr = inverse_transform_predictions(pcat_Temp_components_ATMS,pcat_Temp_mean_ATMS,\
                                                                      pcat_Watr_components_ATMS,pcat_Watr_mean_ATMS,\
                                                                      index_temp_pca,index_watr_pca,watr_pca_scale,Predictions_CrIS)
   if Standardize_TQ:
    predictions_temp = scaler_Temp_ATMS.inverse_transform(predictions_temp)
    predictions_watr = scaler_Watr_ATMS.inverse_transform(predictions_watr)
   predictions_watr = np.exp(predictions_watr)
   Predictions_CrIS = np.c_[predictions_temp[:,:72],predictions_watr,predictions_temp[:,-1]]

   predictions_temp, predictions_watr = inverse_transform_predictions(pcat_Temp_components_ATMS,pcat_Temp_mean_ATMS,\
                                                                      pcat_Watr_components_ATMS,pcat_Watr_mean_ATMS,\
                                                                      index_temp_pca,index_watr_pca,watr_pca_scale,Predictions_ATMS)
   if Standardize_TQ:
    predictions_temp = scaler_Temp_ATMS.inverse_transform(predictions_temp)
    predictions_watr = scaler_Watr_ATMS.inverse_transform(predictions_watr)
   predictions_watr = np.exp(predictions_watr)
   Predictions_ATMS = np.c_[predictions_temp[:,:72],predictions_watr,predictions_temp[:,-1]]
   #Predictions = pcat_ATMS.inverse_transform(Predictions)
   #Predictions_CrIS = pcat_ATMS.inverse_transform(Predictions_CrIS)
   #Predictions_ATMS = pcat_ATMS.inverse_transform(Predictions_ATMS)

 print("Predictions=",Predictions[0,:])
 print("Predictions=",Predictions[10000,:])

 predall = np.zeros((Number_Of_Profiles_sav,Number_Of_EOFs),dtype=Default_Real_Type) + Default_Bad_Value
 predall_cris = np.zeros((Number_Of_Profiles_sav,Number_Of_EOFs),dtype=Default_Real_Type) + Default_Bad_Value
 predall_atms = np.zeros((Number_Of_Profiles_sav,Number_Of_EOFs),dtype=Default_Real_Type) + Default_Bad_Value

 predall_cld  = np.zeros((Number_Of_Profiles_sav,Number_Of_Cloud_EOFs),dtype=Default_Real_Type) + Default_Bad_Value
 predall_emis = np.zeros((Number_Of_Profiles_sav,Number_Of_Emissivity_EOFs),dtype=Default_Real_Type) + Default_Bad_Value
 predall_bias = np.zeros((Number_Of_Profiles_sav,Number_Of_Channels),dtype=Default_Real_Type) + Default_Bad_Value

 # print('Predictions min=',np.nanmin(Predictions,axis=0))
 # print('Predictions max=',np.nanmax(Predictions,axis=0))
 # print('Predictions p50=',np.nanpercentile(Predictions,[50.],axis=0))
 if emistransform: 
#  predictions_emis_cris = pcat_Emis_CrIS.inverse_transform(Predictions_emis[:,index_emis_CrIS_pca])
#  predictions_emis_atms = pcat_Emis_ATMS.inverse_transform(Predictions_emis[:,index_emis_ATMS_pca])

  predictions_emis_cris = pca_inverse_transform(pca_Emis_components_CrIS,pca_Emis_mean_CrIS,Predictions_emis[:,index_emis_CrIS_pca])
  predictions_emis_atms = pca_inverse_transform(pca_Emis_components_ATMS,pca_Emis_mean_ATMS,Predictions_emis[:,index_emis_ATMS_pca])

# if Standardize_Emis:
#   predictions_emis_atms = scalert_Emis_ATMS.inverse_transform(predictions_emis_atms)
#   predictions_emis_cris = scalert_Emis_CrIS.inverse_transform(predictions_emis_cris)

 Predictions_emis = np.c_[predictions_emis_atms,predictions_emis_cris]

 #---move Predictions into target array
 predall[Index_Good_Initial_QC,:] = Predictions
 predall_cris[Index_Good_Initial_QC,:] = Predictions_CrIS
 predall_atms[Index_Good_Initial_QC,:] = Predictions_ATMS
 predall_cld[Index_Good_Initial_QC,:] = Predictions_cloud
 predall_bias[Index_Good_Initial_QC,:] = Predictions_bias
 predall_emis[Index_Good_Initial_QC,:] = Predictions_emis

 Predictions = predall.copy()
 Predictions_CrIS = predall_cris.copy()
 Predictions_ATMS = predall_atms.copy()
 Predictions_cloud = predall_cld.copy()
 Predictions_emis = predall_emis.copy()
 Predictions_bias = predall_bias.copy()

 print('Predictions min=',np.nanmin(Predictions,axis=0))
 print('Predictions max=',np.nanmax(Predictions,axis=0))
 print('Predictions p50=',np.nanpercentile(Predictions,[50.],axis=0))

 #---rescale water vapor profile
 #Predictions[:,Index_Water] = Predictions[:,Index_Water]*water_scale_ATMS
 #Predictions_CrIS[:,Index_Water] = Predictions_CrIS[:,Index_Water]*water_scale_ATMS
 #Predictions_ATMS[:,Index_Water] = Predictions_ATMS[:,Index_Water]*water_scale_ATMS

 #---rescale bias correction
 for ichan in range(0,Number_Of_Channels):
  minb = bias_minmax[ichan,0]
  maxb = bias_minmax[ichan,1]
  Predictions_bias[:,ichan] = (maxb - minb)*Predictions_bias[:,ichan] + minb

 #---rescale cloud parameters
 for iparam in range(0,Number_Of_Cloud_EOFs):
  minb = cloud_minmax[iparam,0]
  maxb = cloud_minmax[iparam,1]
  Predictions_cloud[:,iparam] = (maxb - minb)*Predictions_cloud[:,iparam] + minb
  
 OutputFile_ATMS = OutputFile + '.atms'
 OutputFile_CrIS = OutputFile + '.cris'
 tic_write = time.time()
 print('MIIDAPS-AI RUN: Post time = ', (tic_write - toc)/60)

 write_nc(OutputFile, Predictions, Predictions_bias, Predictions_cloud, Predictions_emis)
 write_nc(OutputFile_ATMS, Predictions_ATMS, Predictions_bias, Predictions_cloud, Predictions_emis)
 write_nc(OutputFile_CrIS, Predictions_CrIS, Predictions_bias, Predictions_cloud, Predictions_emis)

 toc = time.time()
 print('MIIDAPS-AI RUN: Write time = ', (toc - tic_write)/60)
 print('MIIDAPS-AI RUN: Total time = ', (toc - tic)/60)

 return 
 
 #---write Predictions for all cases 
 f = open(OutputFile,'wb')
 fat = open(OutputFile_ATMS,'wb') 
 fcr = open(OutputFile_CrIS,'wb') 
 # with open(OutputFile,'wb') as f:
 # with open(OutputFile_ATMS,'wb') as fat:
 # with open(OutputFile_CrIS,'wb') as fcr:

 for i, p in enumerate(Predictions):
   pb = Predictions_bias[i,:].squeeze()
   pc = Predictions_cloud[i,:].squeeze()
   pe = Predictions_emis[i,:].squeeze()
   p.astype(Default_Real_BigEndian_Type).tofile(f)
   pc.astype(Default_Real_BigEndian_Type).tofile(f)
   pe.astype(Default_Real_BigEndian_Type).tofile(f)
   pb.astype(Default_Real_BigEndian_Type).tofile(f)

   p = Predictions_CrIS[i,:].squeeze()
   p.astype(Default_Real_BigEndian_Type).tofile(fcr)
   pc.astype(Default_Real_BigEndian_Type).tofile(fcr)
   pe.astype(Default_Real_BigEndian_Type).tofile(fcr)
   pb.astype(Default_Real_BigEndian_Type).tofile(fcr)

   p = Predictions_ATMS[i,:].squeeze()
   p.astype(Default_Real_BigEndian_Type).tofile(fat)
   pc.astype(Default_Real_BigEndian_Type).tofile(fat)
   pe.astype(Default_Real_BigEndian_Type).tofile(fat)
   pb.astype(Default_Real_BigEndian_Type).tofile(fat)

 f.close()
 fcr.close()
 fat.close()

 toc = time.time()
 print('MIIDAPS-AI RUN: Write time = ', (toc - tic_write)/60)
 print('MIIDAPS-AI RUN: Total time = ', (toc - tic)/60)

 return 

if __name__ == '__main__':
   
  main(sys.argv[1:])
