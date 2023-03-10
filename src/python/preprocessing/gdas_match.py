import numpy as np
from dump_util import read_dump

class gdas_match:

  def __init__(self,gdas_type,gdas_dump_dir):
     self.gdas_type = gdas_type
     self.init_grid()
     self.gdas_dump_dir = gdas_dump_dir
     #---build gdas latitude, longitudes
     lats_1d = np.linspace(self.y0,self.y1,self.nlat)
     lons_1d = np.linspace(self.x0,self.x1,self.nlon)
     lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
     self.lon_grid1d = lon_grid.reshape(self.nlat*self.nlon)
     self.lat_grid1d = lat_grid.reshape(self.nlat*self.nlon)
     self.xy = np.c_[self.lon_grid1d, self.lat_grid1d]

     iy_1d = np.arange(0,self.nlat)
     ix_1d = np.arange(0,self.nlon)
     ix_grid, iy_grid = np.meshgrid(ix_1d, iy_1d)
     self.ix_grid1d = ix_grid.reshape(self.nlat*self.nlon)
     self.iy_grid1d = iy_grid.reshape(self.nlat*self.nlon)
     self.ind_xy = np.c_[self.ix_grid1d, self.iy_grid1d]
     
  def interp_weights(self,d=2):
    #---compute interpolation weights between two irregular grids ... probably could speed up
    #   because both are regular on this application.
    # ref: 
    # https://stackoverflow.com/questions/20915502/
    #  speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    
    import scipy.interpolate as spint
    import scipy.spatial.qhull as qhull
    #--triangulate points
    tri = qhull.Delaunay(self.xy)
    #---find vertices/simplexes of triangles
    simplex = tri.find_simplex(self.uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = self.uv - temp[:, d]
    #---fancy trick to get barycentric coordinates 
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    #---return vertices and weights for interpolation
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

  def interpolate(self,values):
    #---fancy-schmancy 
    return np.einsum('nj,nj->n', np.take(values, self.vtx), self.wts)

  def interpolate_xy_setup(self,satellite_lat,satellite_lon):

    self.uv = np.c_[satellite_lon, satellite_lat]
    
    #    print "xy_in=",xy.shape, satellite_lon.shape
    #---compute interpolation weights between ECMWF grid and EDF grid 
    vtx, wts = self.interp_weights()
    self.vtx = vtx
    self.wts = wts
    #---compute nearest indexes for padding
    #self.nearest_index(self,satellite_lat,satellite_lon)
    
  def nearest_index(self,satellite_lat,satellite_lon):
    dx = (self.x1 - self.x0)/(self.nlon-1)
    dy = (self.y1 - self.y0)/(self.nlat-1)
    self.nearest_x = ((satellite_lon - self.x0)/dx + 0.49).astype(int)
    self.nearest_y = ((satellite_lat - self.y0)/dy + 0.49).astype(int)
    self.nearest_x[self.nearest_x < 0] = 0
    self.nearest_x[self.nearest_x >= self.nlon] = self.nlon-1
    self.nearest_y[self.nearest_y < 0] = 0
    self.nearest_y[self.nearest_y >= self.nlat] = self.nlat-1

  def interpolate_xy_single(self,variable_in):

    variable_interpolate=self.interpolate(variable_in.flatten())
    return variable_interpolate

  def gdas_read(self):
      
    keys= ['ps','ice','snow','lfrac','tskin']
    ky3d = [False,False,False,False,False]
    read_cloud = True
    gdas_data = read_dump(self.gdas_filename,keys,ky3d,self.nlat,self.nlon,self.nlev,read_cloud=read_cloud,big_endian=False) 
    snow = gdas_data['snow']
    snow[(snow > 5)] = 0.0
    gdas_data['snow'] = snow 
    self.gdas_data = gdas_data
    self.gdas_keys = keys

  def init_grid(self):
 
    if self.gdas_type==0:
      nlat=181; nlon=360 
      x0=0; x1=359.0
      y0=-90; y1=90
      y0=90; y1=-90
      gdas_resolution = "1p00"
    elif self.gdas_type==1:
      nlat=721; nlon=1440
      x0=0; x1=359.75
      y0=-90; y1=90
      y0=90; y1=-90
      gdas_resolution = "0p25"
    elif self.gdas_type==2:
      nlat=181; nlon=360 
      x0=0; x1=359.0
      y0=-90; y1=90
      gdas_resolution=None
    else:
      nlat=None; nlon=None
      x0=None; x1=None
      y0=None; y1=None
      gdas_resolution=None
    self.nlat = nlat
    self.nlon = nlon
    self.x0   = x0
    self.x1   = x1 
    self.y0   = y0
    self.y1   = y1
    self.nlev = 26
    self.gdas_resolution = gdas_resolution

  def get_gdas_fileinfo(self,satellite_year,satellite_month,satellite_day,satellite_hour):
    from datetime import datetime
    from datetime import timedelta
    satellite_date = datetime(satellite_year,satellite_month,satellite_day,satellite_hour)
    self.gdas_date = satellite_date
    if ((satellite_hour >= 21) | (satellite_hour < 3)):
      self.gdas_hour = "00"
      if (satellite_hour >= 21 ): self.gdas_date = satellite_date + timedelta(days=1)
    elif ((satellite_hour >= 3) & (satellite_hour < 9)):
      self.gdas_hour = "06"
    elif ((satellite_hour >= 9) & (satellite_hour < 15)):
      self.gdas_hour = "12"
    elif ((satellite_hour >= 15) & (satellite_hour < 21)):
      self.gdas_hour = "18"
   
    self.gdas_day = self.gdas_date.strftime("%d")
    self.gdas_year = self.gdas_date.strftime("%Y")
    self.gdas_month = self.gdas_date.strftime("%m")
    self.gdas_filename = "%s/%s%s%s/gdas.t%sz.pgrb2.%s.f000.bin" % (self.gdas_dump_dir,self.gdas_year,self.gdas_month,self.gdas_day,self.gdas_hour,self.gdas_resolution)

     
