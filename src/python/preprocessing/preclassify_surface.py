import numpy as np

def applySeaIceClimo(month,xlat,xlon,landindex):

    nLonBins=36
    nTimeBins=12
    WidthLonBin=10. # width of longitude bins for storing latitude thresholds (degrees)

    oc_typ = 0

    threshLatNH = np.zeros((nTimeBins,nLonBins),dtype=np.float32)
    threshLatSH = np.zeros((nTimeBins,nLonBins),dtype=np.float32)
    
    threshLatNH[0,:] = [72.3,59.5,59.5,65.3,66.0,66.2,65.0,65.0,65.0,65.0,65.0,65.0,65.0,45.8,45.8,49.1,52.5,56.7, \
                        52.5,52.5,56.7,56.7,56.7,50.0,50.0,50.0,50.0,50.0,50.0,45.8,48.3,58.0,65.0,67.0,67.0,69.3]

    threshLatNH[1,:] = [72.0,57.0,57.0,64.0,64.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,42.0,42.0,46.0,50.0,55.0, \
                        50.0,50.0,55.0,55.0,55.0,50.0,50.0,50.0,50.0,50.0,50.0,45.0,48.0,58.0,65.0,67.0,67.0,69.0]

    threshLatNH[2,:] = [72.3,59.5,59.5,65.3,66.0,66.2,65.0,65.0,65.0,65.0,65.0,65.0,65.0,45.8,45.8,49.1,52.5,56.7, \
                        52.5,52.5,56.7,56.7,56.7,50.0,50.0,50.0,50.0,50.0,50.0,45.8,48.3,58.0,65.0,67.0,67.0,69.3]

    threshLatNH[3,:] = [72.7,62.0,62.0,66.6,68.0,67.3,65.0,65.0,65.0,65.0,65.0,65.0,65.0,49.6,49.6,52.3,55.0,58.3, \
                        55.0,55.0,58.3,58.3,58.3,50.0,50.0,50.0,50.0,50.0,50.0,46.7,48.7,58.0,65.0,67.0,67.0,69.7]

    threshLatNH[4,:] = [73.0,64.5,64.5,68.0,70.0,68.5,65.0,65.0,65.0,65.0,65.0,65.0,65.0,53.5,53.5,55.5,57.5,60.0, \
                        57.5,57.5,60.0,60.0,60.0,50.0,50.0,50.0,50.0,50.0,50.0,47.5,49.0,58.0,65.0,67.0,67.0,70.0]

    threshLatNH[5,:] = [73.3,67.0,67.0,69.3,72.0,69.7,65.0,65.0,65.0,65.0,65.0,65.0,65.0,53.5,57.3,58.6,60.0,61.7, \
                        60.0,60.0,61.7,61.7,61.7,50.0,50.0,50.0,50.0,50.0,50.0,48.3,49.3,58.0,65.0,67.0,67.0,70.3]

    threshLatNH[6,:] = [73.7,69.5,69.5,70.6,74.0,70.8,65.0,65.0,65.0,65.0,65.0,65.0,65.0,53.5,61.1,61.8,62.5,63.3, \
                        62.5,62.5,63.3,63.3,63.3,50.0,50.0,50.0,50.0,50.0,50.0,49.2,49.7,58.0,65.0,67.0,67.0,70.7]

    threshLatNH[7,:] = [74.0,72.0,72.0,72.0,76.0,72.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0, \
                        65.0,65.0,65.0,65.0,65.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,58.0,65.0,67.0,67.0,71.0]

    threshLatNH[8,:] = [73.7,69.5,69.5,70.6,74.0,70.8,65.0,65.0,65.0,65.0,65.0,65.0,65.0,53.5,61.1,61.8,62.5,63.3, \
                        62.5,62.5,63.3,63.3,63.3,50.0,50.0,50.0,50.0,50.0,50.0,49.2,49.7,58.0,65.0,67.0,67.0,70.7]

    threshLatNH[9,:] = [73.3,67.0,67.0,69.3,72.0,69.7,65.0,65.0,65.0,65.0,65.0,65.0,65.0,53.5,57.3,58.6,60.0,61.7, \
                        60.0,60.0,61.7,61.7,61.7,50.0,50.0,50.0,50.0,50.0,50.0,48.3,49.3,58.0,65.0,67.0,67.0,70.3]

    threshLatNH[10,:] = [73.0,64.5,64.5,68.0,70.0,68.5,65.0,65.0,65.0,65.0,65.0,65.0,65.0,53.5,53.5,55.5,57.5,60.0, \
                        57.5,57.5,60.0,60.0,60.0,50.0,50.0,50.0,50.0,50.0,50.0,47.5,49.0,58.0,65.0,67.0,67.0,70.0]

    threshLatNH[11,:] = [72.7,62.0,62.0,66.6,68.0,67.3,65.0,65.0,65.0,65.0,65.0,65.0,65.0,49.6,49.6,52.3,55.0,58.3, \
                        55.0,55.0,58.3,58.3,58.3,50.0,50.0,50.0,50.0,50.0,50.0,46.7,48.7,58.0,65.0,67.0,67.0,69.7]

    #---Southern Hem. values (note: these are absolute values of latitude)

    threshLatSH[0,:] = [62.5,62.8,63.3,60.8,60.0,59.2,59.2,59.2,59.2,59.2,59.2,59.2,59.2,59.2,59.2,59.2,62.3,62.3, \
                        62.3,62.3,62.3,62.3,62.3,62.3,63.3,63.3,61.7,60.0,58.8,58.8,57.7,57.5,58.6,58.6,62.5,62.5]

    threshLatSH[1,:] = [65.0,65.0,65.0,62.0,61.0,60.0,60.0,60.0,60.0,60.0,60.0,60.0,60.0,60.0,60.0,60.0,64.0,64.0, \
                        64.0,64.0,64.0,64.0,64.0,64.0,64.0,64.0,62.0,60.0,59.0,59.0,58.0,58.0,60.0,62.0,65.0,65.0]

    threshLatSH[2,:] = [62.5,62.8,63.3,60.8,60.0,59.2,59.2,59.2,59.2,59.2,59.2,59.2,59.2,59.2,59.2,59.2,62.3,62.3, \
                        62.3,62.3,62.3,62.3,62.3,62.3,63.3,63.3,61.7,60.0,58.8,58.8,57.7,57.5,58.6,58.6,62.5,62.5]

    threshLatSH[3,:] = [60.0,60.6,61.7,59.7,59.0,58.3,58.3,58.3,58.3,58.3,58.3,58.3,58.3,58.3,58.3,58.3,60.7,60.7, \
                        60.7,60.7,60.7,60.7,60.7,60.7,62.7,62.7,61.3,60.0,58.7,58.7,57.3,57.0,57.3,57.3,60.0,60.0]

    threshLatSH[4,:] = [57.5,58.5,60.0,58.5,58.0,57.5,57.5,57.5,57.5,57.5,57.5,57.5,57.5,57.5,57.5,57.5,59.0,59.0, \
                        59.0,59.0,59.0,59.0,59.0,59.0,62.0,62.0,61.0,60.0,58.5,58.5,57.0,56.5,56.0,56.0,57.5,57.5]

    threshLatSH[5,:] = [55.0,56.3,58.3,57.3,57.0,56.7,56.7,56.7,56.7,56.7,56.7,56.7,56.7,56.7,56.7,56.7,57.3,57.3, \
                        57.3,57.3,57.3,57.3,57.3,57.3,61.3,61.3,56.1,60.0,58.3,58.3,56.7,56.0,54.6,54.0,55.0,55.0]

    threshLatSH[6,:] = [52.5,54.1,56.7,56.2,56.0,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.7,55.7, \
                        55.7,55.7,55.7,55.7,55.7,55.7,57.5,57.5,55.6,60.0,58.2,58.2,56.3,55.5,53.3,52.0,52.5,52.5]

    threshLatSH[7,:] = [50.0,52.0,55.0,55.0,55.0,55.0,55.0,55.0,55.0,55.0,55.0,55.0,55.0,55.0,55.0,55.0,54.0,54.0, \
                        54.0,54.0,54.0,54.0,54.0,54.0,60.0,60.0,60.0,60.0,58.0,58.0,56.0,55.0,52.0,50.0,50.0,50.0]

    threshLatSH[8,:] = [52.5,54.1,56.7,56.2,56.0,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.8,55.7,55.7, \
                        55.7,55.7,55.7,55.7,55.7,55.7,57.5,57.5,55.6,60.0,58.2,58.2,56.3,55.5,53.3,52.0,52.5,52.5]

    threshLatSH[9,:] = [55.0,56.3,58.3,57.3,57.0,56.7,56.7,56.7,56.7,56.7,56.7,56.7,56.7,56.7,56.7,56.7,57.3,57.3, \
                        57.3,57.3,57.3,57.3,57.3,57.3,61.3,61.3,56.1,60.0,58.3,58.3,56.7,56.0,54.6,54.0,55.0,55.0]

    threshLatSH[10,:] = [57.5,58.5,60.0,58.5,58.0,57.5,57.5,57.5,57.5,57.5,57.5,57.5,57.5,57.5,57.5,57.5,59.0,59.0, \
                        59.0,59.0,59.0,59.0,59.0,59.0,62.0,62.0,61.0,60.0,58.5,58.5,57.0,56.5,56.0,56.0,57.5,57.5]

    threshLatSH[11,:] = [60.0,60.6,61.7,59.7,59.0,58.3,58.3,58.3,58.3,58.3,58.3,58.3,58.3,58.3,58.3,58.3,60.7,60.7, \
                        60.7,60.7,60.7,60.7,60.7,60.7,62.7,62.7,61.3,60.0,58.7,58.7,57.3,57.0,57.3,57.3,60.0,60.0]

    surface_type=landindex
    
    month_index = int(month-1)
    #---Only apply if landindex=0 (ocean flag from database)
    if (landindex == oc_typ):
       #---Convert longitude to index (account for lons -180 to +180)
       #---Index used to select proper element from climatology
       if(xlon < 0.):
          iLonBin=int((xlon+360.)/WidthLonBin)
       else:
          iLonBin=int(xlon/WidthLonBin)
       if(iLonBin >= nLonBins): iLonBin=nLonBins-1
       if(iLonBin < 0): iLonBin=0

       #---Check hemisphere, and use appropriate climatology
       if(xlat >= 0.):
          if(np.abs(xlat) < threshLatNH[month_index,iLonBin]): surface_type=oc_typ
       else:
          if(np.abs(xlat) < threshLatSH[month_index,iLonBin]): surface_type=oc_typ

    return surface_type

def preclassify_surface(year,month,taw,lat,lon,landindex,TskPreclass):
   SILOW = 5.0; SIHIGH = 10.0
   LATTH1 = 30.0; LATTH2 = 40.0; LATTH3 = 50.0               
   #---classify land, ocean, sea ice, snow based on GDAS surface parameters
   OC_TYP = 0; SEAICE_TYP = 1; LD_TYP = 2; SNOW_TYP = 3
   
   coe_oc   = np.zeros((2,7),dtype=np.float32)
   coe_land = np.zeros((2,7),dtype=np.float32)

   #---Fitting coefficients to predict ta92 over open ocean 
   coe_oc[0,:] = [6.76185e2,  2.55301e0,  2.44504e-1, -6.88612e0,   \
                  -5.01409e-3, -1.41372e-3,  1.59245e-2]

   #---Fitting coefficients to predict ta157 over open ocean                       
   coe_oc[1,:] = [5.14546e2,  6.06543e0, -6.09327e0, -2.81614e0,   \
                  -1.35415e-2,  1.29683e-2 , 7.69443e-3]

   #---Fitting coefficients to predict ta92 over snow-free land
   coe_land[0,:] = [-3.09009e2,  1.74746e0, -2.01890e0,  3.43417e0, \
                    -2.85680e-3,  3.53140e-3, -4.39255e-3]

   #---Fitting coefficients to predict ta157 over snow-free land
   coe_land[1,:] = [-1.01014e1,  3.97994e0, -4.11268e0,  1.26658e0, \
                    -9.52526e-3,  8.75558e-3,  4.77981e-4]
                                     
   TA_ICE = np.zeros((3),dtype=np.float32)
   TA_SNOW = np.zeros((3),dtype=np.float32)

   #---Predict SEA ICE TA92 and TA157 from TA23 ~ TA50 using open ocean fitting coeffs.
   for nd in range(0,1):
       TA_ICE[nd]= coe_oc[nd,0]
       for ich in range(0,3):
          TA_ICE[nd] = TA_ICE[nd] + coe_oc[nd,ich+1]*taw[ich]
       for ich in range(0,3):
          TA_ICE[nd] = TA_ICE[nd] + coe_oc[nd,ich+4]*taw[ich]*taw[ich]
   TA92_SICE  = TA_ICE[0]
   TA157_SICE = TA_ICE[1]
   #---Predict SEA ICE TA92 and TA157 from TA23 ~ TA50 using open ocean fitting coeffs.
   for nd in range(0,1):
      TA_SNOW[nd]= coe_land[nd,0]
      for ich in range(0,3):
         TA_SNOW[nd] = TA_SNOW[nd] + coe_land[nd,ich+1]*taw[ich]
      for ich in range(0,3):
         TA_SNOW[nd] = TA_SNOW[nd] + coe_land[nd,ich+4]*taw[ich]*taw[ich]
   TA92_SNOW  = TA_SNOW[0]
   TA157_SNOW = TA_SNOW[1]
   #---COMPUTE SI = TA23 - TA92
   SI = taw[0]-taw[3]
   #---Predict surface types
   if (landindex == OC_TYP):    # over ocean conditions
      surface_type = OC_TYP
      dt_1 = TA92_SICE - taw[3]
      if ( (dt_1 >= 10.0) & (abs(lat) >= LATTH2)):  surface_type = SEAICE_TYP
      if ( (abs(lat) >= LATTH2) & (SI >= SIHIGH)): surface_type = SEAICE_TYP
      if ( (abs(lat) >= LATTH3) & (SI >= SILOW)): surface_type = SEAICE_TYP
      if ( (abs(lat) >= LATTH3) & (taw[0] >= 235.0)): surface_type = SEAICE_TYP
      #---Get help from the tskin temperature
      if (TskPreclass >= 280.): surface_type = OC_TYP 
      if ( (TskPreclass <= 265.) & (TskPreclass >=0.)): surface_type = SEAICE_TYP 
      #---Get help from the latitude
      if (abs(lat) <= 50.): surface_type = OC_TYP
      #---Get help from the latitude and longitude 
#      call applySeaIceClimo(Year, Julday, lat, lon, landindex, surface_type)
      surface_type = applySeaIceClimo(month,lat,lon,surface_type)

   else:                             # over land conditions
      surface_type = LD_TYP
      dt_1 = TA92_SNOW-taw[3]
      if( (dt_1 >=10.0) & (taw[0] <= 260.0) & (abs(lat) >= LATTH1) ): surface_type = SNOW_TYP
      if( (abs(lat) >= LATTH2) & (SI >= SIHIGH) & (taw[0] <= 260.0) ): surface_type = SNOW_TYP
      if( (abs(lat) >= LATTH3) & (SI >= SILOW) & (taw[0] <= 260.0)): surface_type = SNOW_TYP
      if( (taw[0] <= 210.0) & (taw[2] <= 225.0) & (abs(lat) >= LATTH3)): surface_type = SNOW_TYP
      #---Get help from the tskin temperature
      if (TskPreclass >= 280.): surface_type = LD_TYP
      if ( (TskPreclass <= 265.) & (TskPreclass >=0.) ): surface_type = SNOW_TYP 

   return surface_type

