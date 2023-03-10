# MIIDAPS-AI

## Table of contents
* [General Info](#general-info)
* [Inputs](#inputs)
* [Languages](#languages)
* [To Run](#to-run)
* [Lookup Tables](#lookup-tables)
* [Contact](#contact)

## General Info 

MIIDAPS-AI is a software package to perform a non-linear mapping between NOAA-20 CrIS/ATMS brightness temperature
observations and geophysical parameters.  Current valid products are temperature and moisture profiles between the 
surface and 0.005hPa and surface temperature.  Spectra emissivity and cloud parameters (IWP, CLW) are also produced, but
have not been extensively validated.

MIIDAPS-AI is built in Python and uses Keras and TensorFlow.

## Inputs

Satellite inputs are the 2211 channel CrIS/co-located 22 channel ATMS NetCDFs from the NUCAPS system. Those are the files that have 
one ATMS Field-of-View (FOV) for all channels per 3x3 CrIS Field-Of-Regard (FOR).  On our systems they are named: 
```
NUCAPS-ALL-HR_v2r0_j01_s202004221255599_e202004221256297_c202004221329150.nc
```

Ancillary inputs are GDAS GRB2 analysis 03, 06, 09, forecasts at 0.25 or 1.0 degree resolution.  Those have expected 
filenames

```
gdas.t[SynopticTime]z.pgrb2.[Resolution].[ForecastLength]
```

## Languages

MIIDAPS-AI is mostly written in Python 3.6, but is also compatible with older versions of the software.  I've included 
a YAML file (```py3_miidaps-ai.yml```) which has the Python Conda environment/package versions listed.  A short list of packages includes:
```tensorflow, keras, hdf, netcdf, and numpy```

MIIDAPS-AI run scripts are written in bash and have been tested on standalone Linux servers and clusters (SLURM) on both 
CentOS 6 and 7.7; however CentOS 7.7 is preferred with the package being delivered.  Cluster submission versus command line
submission can be controlled in the ```setup/paths.setup``` configuration file.

MIIDAPS-AI preprocessing of forecast information requires ```wgrib2``` to dump the GDAS grib2 forecasts/analysis files

## To Run 

First download and install Anaconda Python 3.  You can create a Conda environment
based on my YAML as follows:

```
$ ./anaconda3/bin/conda env create --file miidaps-ai/setup/py3_miidaps-ai.yml
```

To test the code with test data.  Modify setup/paths.setup for your machine.  

```
$ cd miidaps-ai/scripts/run_test/
$ test.sh YYYY MM DD
```

## Lookup tables

Total lookup tables : 9 files, (4.6MB)

MIIDAPS-AI lookup tables for channel selection (currently fixed for package being delivered) are formatted ascii files (2 files). 
Any algorithm improvements for currently produced products will be achieved through updates to the following tables.

```
ai_fix_data/v001_431lwmid/ (65K)
```

MIIDAPS-AI model data are in 

```
ai_model_data/v001_431lwmid/ (4.5M)
```

Keras model lookup tables (2 files):

Model weights are in hdf5 format -

```
model_weights.h5
``` 

Model architecture is in .json file format -

```
model.json 
```

Data preprocessing, normalization, etc. are in json format (5 files) 

PCA coefficients - 

```
pca_[instrument]_[edrtype].json
```

normalization - 

```
minmax.json
```






