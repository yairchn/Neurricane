import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import netCDF4 as nc
from generate_TC_fields import TC_fields
from scipy.interpolate import interp1d
# command line:
# python WRF_sampler.py    wrf_path
# python WRF_sampler.py    /Volumes/TimeMachine/hurricanes/Patricia_for_Yair/wrfout_d04_2015-10-22_16_00_00.nc

# The model recieves 2 pathes, one for the wrf_data and one for the storm cengter location
# storm center is taken from https://www.aoml.noaa.gov/hrd/Storm_pages/
# the code returns the fields interpolated to pressure surfaces so they can be used to compute PGF as gradfiuent if Z in the horizontal direction on constant pressure surfaces

def WRF_fields(wrf_path):

	data = nc.Dataset(wrf_path, 'r')
	LON = np.squeeze(data.variables['XLONG'])
	LAT = np.squeeze(data.variables['XLAT'])
	day =   np.float(wrf_path[-14:-12])
	year =  np.float(wrf_path[-22:-18])
	month = np.float(wrf_path[-17:-15])
	hour =  np.float(wrf_path[-11:-9])

	P = np.squeeze(np.add(data.variables['PB'], data.variables['P']))
	P00 = np.array(data.variables['P00'])
	P_new = np.linspace(85000.0, 2000.0, num = 100)

	# convert the Lat Lon coordinates to X,Y [km] respectively to storm center location
	ind = np.unravel_index(np.argmin(P[2,:,:], axis=None), P[1,:,:].shape)
	x_center = LON[ind]
	y_center = LAT[ind]
	p_center = P[2,ind[0],ind[1]]
	y = np.multiply(111100.0, np.subtract(LAT, y_center))
	x = np.multiply(111100.0, np.multiply(np.subtract(LON, x_center), np.cos(np.deg2rad(LAT))))

	logP = np.squeeze(np.log(P))
	logP_new = np.log(P_new)
	P_out = P_new/100.0

	PH_raw = np.squeeze(np.add(data.variables['PHB'], data.variables['PH']))
	PH_ = np.divide(np.add(PH_raw[0:42:,:],PH_raw[1:43,:,:]),2.0)

	Theta = np.squeeze(np.add(data.variables['T00'], data.variables['T']))
	T_ = np.multiply(Theta,np.power(np.divide(P,100000.0), 287.0/1004.0))
	u_ = np.squeeze(data.variables['U'])
	v_ = np.squeeze(data.variables['V'])

	x0 = len(PH_[0,:,0])
	y0 = len(PH_[0,0,:])

	Z1 = np.zeros((np.shape(P_new)[0],np.shape(PH_[0,:,0])[0],np.shape(PH_[0,0,:])[0]))
	T1 = np.zeros((np.shape(P_new)[0],np.shape(T_[0,:,0])[0], np.shape(T_[0,0,:]) [0]))
	u1 = np.zeros((np.shape(P_new)[0],np.shape(u_[0,:,0])[0], np.shape(u_[0,0,:]) [0]))
	v1 = np.zeros((np.shape(P_new)[0],np.shape(v_[0,:,0])[0], np.shape(v_[0,0,:]) [0]))

	for i in range(x0):
		for j in range(y0):
			f_PH = interp1d(logP[:,i,j], PH_[:,i,j], axis=-1)
			f_T  = interp1d(logP[:,i,j], T_[:,i,j],  axis=-1)
			f_u  = interp1d(logP[:,i,j], u_[:,i,j],  axis=-1)
			f_v  = interp1d(logP[:,i,j], v_[:,i,j],  axis=-1)

			Z1[:,i,j] = f_PH(logP_new)
			T1[:,i,j] = f_T(logP_new)
			u1[:,i,j] = f_u(logP_new)
			v1[:,i,j] = f_v(logP_new)

	# reshape to [x,y,z]
	Z = np.moveaxis(Z1, 0, -1)
	T = np.moveaxis(T1, 0, -1)
	u = np.moveaxis(u1, 0, -1)
	v = np.moveaxis(v1, 0, -1)
	i850 = 0
	i150 = 75
	return 0.0, 0.0, x, y, P_out, Z, T, i850, i150


def WRF_channel_fields(wrf_path):

	data = nc.Dataset(wrf_path, 'r')
	LON  = np.squeeze(data.variables['lon'])
	LAT  = np.squeeze(data.variables['lat'])
	GHT  = np.squeeze(data.variables['ght'])
	temp    = np.squeeze(data.variables['temp'])
	U    = np.squeeze(data.variables['ugrdr'])
	V    = np.squeeze(data.variables['vgrdr'])
	P    = np.squeeze(data.variables['lev']) # p[6] = 850mb

	GHT[np.abs(GHT)>9.9900000e+07]=np.nan
	temp[np.abs(temp)>9.9900000e+07]=np.nan
	U[np.abs(U)>9.9900000e+07]=np.nan
	V[np.abs(V)>9.9900000e+07]=np.nan
	P[np.abs(P)>9.9900000e+07]=np.nan

	ght = np.moveaxis(GHT, 0, -1)
	T = np.moveaxis(GHT, 0, -1)
	u = np.moveaxis(U, 0, -1)
	v = np.moveaxis(V, 0, -1)

	# storm is located at 200,200
	x_center = LON[200]
	y_center = LAT[200]
	ght_center = ght[200,200,6]
	Y = np.multiply(111100.0, np.subtract(LAT, y_center))
	X = np.multiply(111100.0, np.multiply(np.subtract(LON, x_center), np.cos(np.deg2rad(LAT))))
	x, y = np.meshgrid(X, Y, sparse=False, indexing='ij')
	plt.figure('lower  level Z')
	plt.contour(ght[:,:,6])
	plt.show()

	return 0.0, 0.0, x, y, P, ght, T, 15,6
