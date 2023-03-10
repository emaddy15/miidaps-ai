import numpy as np 

def write_ai(filename,nprof,nchan,nEOF,nAnc,nEDRs,npredictor,predictors,targets,tscale,logflag,sfctype,longitude,latitude,time,psurf,nsensors,sensor_ids,sensor_nchan,**kwargs):

 ibe = np.dtype('>i4')
 fbe = np.dtype('>f4')
 endian = kwargs.get('input_endian','big')
 if endian == 'little':
  ibe = np.dtype('<i4')
  fbe = np.dtype('<f4')

# print(endian,fbe,ibe)                                                                                                                                                                        
 s15 = np.dtype('S15')
 f    =  open(filename,'wb')

 xx = np.array([nprof, nchan, nEOF, nAnc, nEDRs])#  = np.fromfile(f,ibe,count=5)                                                                                                               
 print("xx = ", xx)
 xx.astype(ibe).tofile(f)

 #---write sensor info                                                                                                                                                                         
 x = np.array(nsensors)
 x.astype(ibe).tofile(f)
 for isen in range(0,nsensors):
  sensor_nchan[isen].astype(ibe).tofile(f)
  np.array(sensor_ids[isen]).astype(s15).tofile(f)

 logflag.astype(ibe).tofile(f)
 tscale.astype(fbe).tofile(f)

 for ip in range(0,nprof):
   pred = predictors[ip,:].squeeze()     
   pred.astype(fbe).tofile(f)

   targ = targets[ip,:].squeeze() 
   targ.astype(fbe).tofile(f)
                                         
 sfctype.astype(fbe).tofile(f)
 longitude.astype(fbe).tofile(f)
 latitude.astype(fbe).tofile(f)
 time.astype(fbe).tofile(f)
 psurf.astype(fbe).tofile(f)
 f.close()

def write_aiemis(filename,nprof,nchan,nEOF,nAnc,nEDRs,npredictor,predictors,targets,tscale,logflag,sfctype,longitude,latitude,\
                 time,psurf,tsurf,nsensors,sensor_ids,sensor_nchan,**kwargs):

 ibe = np.dtype('>i4')
 fbe = np.dtype('>f4')
 endian = kwargs.get('input_endian','big')
 if endian == 'little':
  ibe = np.dtype('<i4')
  fbe = np.dtype('<f4')

# print(endian,fbe,ibe)                                                                                                                                                                        
 s15 = np.dtype('S15')
 f    =  open(filename,'wb')

 xx = np.array([nprof, nchan, nEOF, nAnc, nEDRs])#  = np.fromfile(f,ibe,count=5)                                                                                                               
 print("xx = ", xx)
 xx.astype(ibe).tofile(f)

 #---write sensor info                                                                                                                                                                         
 x = np.array(nsensors)
 x.astype(ibe).tofile(f)
 for isen in range(0,nsensors):
  sensor_nchan[isen].astype(ibe).tofile(f)
  np.array(sensor_ids[isen]).astype(s15).tofile(f)

 logflag.astype(ibe).tofile(f)
 tscale.astype(fbe).tofile(f)

 for ip in range(0,nprof):
   pred = predictors[ip,:].squeeze()     
   pred.astype(fbe).tofile(f)

   targ = targets[ip,:].squeeze() 
   targ.astype(fbe).tofile(f)
                                         
 sfctype.astype(fbe).tofile(f)
 longitude.astype(fbe).tofile(f)
 latitude.astype(fbe).tofile(f)
 time.astype(fbe).tofile(f)
 psurf.astype(fbe).tofile(f)
 tsurf.astype(fbe).tofile(f)
 f.close()

def read_ai(filename,**kwargs):
 ibe = np.dtype('>i4')
 fbe = np.dtype('>f4')
 nosheader = kwargs.get('nosheader',False)
 cnvdbl    = kwargs.get('cnvdbl',True)
 endian = kwargs.get('input_endian','big')
 if endian == 'little':
  ibe = np.dtype('<i4')
  fbe = np.dtype('<f4')
  
 print(endian,fbe,ibe)
 s15 = np.dtype('S15')
 f    =  open(filename,'rb')
 
 nprof, nchan, nEOF, nAnc, nEDRs  = np.fromfile(f,ibe,count=5)

 #---sensors and numbers of channels per sensor in input file 
 if (nosheader == False):
  nsensors = np.fromfile(f,ibe,count=1)
  sensor_ids = []
  sensor_nchan = np.zeros((nsensors),dtype=np.int)
  for isen in range(0,nsensors[0]):
   nc = np.fromfile(f,ibe,count=1)
   sensor_nchan[isen] = nc 
   ss = np.fromfile(f,s15,count=1)
   sensor_ids.append(ss)
 else:   
  nsensors = 1
  sensor_ids = []
  sensor_nchan = np.zeros((nsensors),dtype=np.int)

 print(nprof, nchan, nEOF, nAnc, nEDRs)
 print(sensor_ids)
 
 npredictor = nchan + nAnc
 predictors = np.zeros((nprof,npredictor),dtype=np.float32)
 logflag = np.fromfile(f,ibe,count=nEOF)
 tscale = np.fromfile(f,fbe,count=nEOF)
 targets = np.zeros((nprof,nEOF),dtype=np.float32)
 for ip in range(0,nprof):
#   print ip, nprof
   pred = np.fromfile(f,fbe,count=npredictor)
   targ = np.fromfile(f,fbe,count=nEOF)
   predictors[ip,:] = pred
   targets[ip,:]    = targ

 sfctype   = np.fromfile(f,fbe,count=nprof)
 longitude = np.fromfile(f,fbe,count=nprof)
 latitude  = np.fromfile(f,fbe,count=nprof)
 time      = np.fromfile(f,fbe,count=nprof)
 psurf     = np.fromfile(f,fbe,count=nprof)
 if cnvdbl == True:
   predictors = np.float64(predictors)
   targets    = np.float64(targets)
 return nprof, nchan, nEOF, nAnc, nEDRs, npredictor, predictors, targets, tscale, logflag, sfctype, longitude, latitude, time, psurf, \
     nsensors, sensor_ids, sensor_nchan

def read_aiemis(filename,**kwargs):
 ibe = np.dtype('>i4')
 fbe = np.dtype('>f4')
 cnvdbl    = kwargs.get('cnvdbl',True)
 endian = kwargs.get('input_endian','big')
 if endian == 'little':
  ibe = np.dtype('<i4')
  fbe = np.dtype('<f4')
  
 print(endian,fbe,ibe)
 s15 = np.dtype('S15')
 f    =  open(filename,'rb')
 
 nprof, nchan, nEOF, nAnc, nEDRs  = np.fromfile(f,ibe,count=5)

 #---sensors and numbers of channels per sensor in input file 
 nsensors = np.fromfile(f,ibe,count=1)
 sensor_ids = []
 sensor_nchan = np.zeros((nsensors),dtype=np.int)
 for isen in range(0,nsensors):
  nc = np.fromfile(f,ibe,count=1)
  sensor_nchan[isen] = nc 
  ss = np.fromfile(f,s15,count=1)
  sensor_ids.append(ss)
 
 print(nprof, nchan, nEOF, nAnc, nEDRs)
 print(sensor_ids)
 
 npredictor = nchan + nAnc
 predictors = np.zeros((nprof,npredictor),dtype=np.float32)
 logflag = np.fromfile(f,ibe,count=nEOF)
 tscale = np.fromfile(f,fbe,count=nEOF)
 targets = np.zeros((nprof,nEOF),dtype=np.float32)
 for ip in range(0,nprof):
   pred = np.fromfile(f,fbe,count=npredictor)
   targ = np.fromfile(f,fbe,count=nEOF)
   predictors[ip,:] = pred
   targets[ip,:]    = targ

 sfctype   = np.fromfile(f,fbe,count=nprof)
 longitude = np.fromfile(f,fbe,count=nprof)
 latitude  = np.fromfile(f,fbe,count=nprof)
 time      = np.fromfile(f,fbe,count=nprof)
 psurf     = np.fromfile(f,fbe,count=nprof)
 tsurf     = np.fromfile(f,fbe,count=nprof)
 if cnvdbl == True:
   predictors = np.float64(predictors)
   targets    = np.float64(targets)
 return nprof, nchan, nEOF, nAnc, nEDRs, npredictor, predictors, targets, tscale, logflag, sfctype, longitude, latitude, time, psurf, tsurf, \
     nsensors, sensor_ids, sensor_nchan

def read_aibkg(filename,**kwargs):
 ibe = np.dtype('>i4')
 fbe = np.dtype('>f4')
 cnvdbl    = kwargs.get('cnvdbl',True)
 endian = kwargs.get('input_endian','big')
 if endian == 'little':
  ibe = np.dtype('<i4')
  fbe = np.dtype('<f4')
  
 print(endian,fbe,ibe)
 s15 = np.dtype('S15')
 f    =  open(filename,'rb')
 
 nprof, nchan, nEOF, nAnc, nEDRs  = np.fromfile(f,ibe,count=5)

 #---sensors and numbers of channels per sensor in input file 
 nsensors = np.fromfile(f,ibe,count=1)
 sensor_ids = []
 sensor_nchan = np.zeros((nsensors),dtype=np.int)
 for isen in range(0,nsensors):
  nc = np.fromfile(f,ibe,count=1)
  sensor_nchan[isen] = nc 
  ss = np.fromfile(f,s15,count=1)
  sensor_ids.append(ss)
 
 print(nprof, nchan, nEOF, nAnc, nEDRs)
 print(sensor_ids)
 
 npredictor = nchan + nAnc
 predictors = np.zeros((nprof,npredictor),dtype=np.float32)
 logflag    = np.fromfile(f,ibe,count=nEOF)
 tscale     = np.fromfile(f,fbe,count=nEOF)
 targets    = np.zeros((nprof,nEOF),dtype=np.float32)
 bkgtargets = np.zeros((nprof,nEOF),dtype=np.float32)
 for ip in range(0,nprof):
   pred  = np.fromfile(f,fbe,count=npredictor)
   targb = np.fromfile(f,fbe,count=nEOF)
   targ  = np.fromfile(f,fbe,count=nEOF)
   predictors[ip,:] = pred
   targets[ip,:]    = targ
   bkgtargets[ip,:] = targb

 sfctype   = np.fromfile(f,fbe,count=nprof)
 longitude = np.fromfile(f,fbe,count=nprof)
 latitude  = np.fromfile(f,fbe,count=nprof)
 time      = np.fromfile(f,fbe,count=nprof)
 psurf     = np.fromfile(f,fbe,count=nprof)
 if cnvdbl == True:
   predictors = np.float64(predictors)
   targets    = np.float64(targets)
 return nprof, nchan, nEOF, nAnc, nEDRs, npredictor, predictors, targets, bkgtargets, tscale, logflag, \
  sfctype, longitude, latitude, time, psurf, \
  nsensors, sensor_ids, sensor_nchan
