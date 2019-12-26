import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import netCDF4 as nc
from generate_TC_fields import TC_fields
from scipy.interpolate import interp1d
# command line:
# python WRF_sampler.py    wrf_path                                                                                     center_path
# python WRF_sampler.py    /Volumes/TimeMachine/hurricanes/Patricia_for_Yair/wrfout_d04_2015-10-22_16_00_00.nc      /Users/yaircohen/Documents/codes/Neurricane/storm_center/Patricia_trajectory.nc

# The model recieves 2 pathes, one for the wrf_data and one for the storm cengter location
# storm center is taken from https://www.aoml.noaa.gov/hrd/Storm_pages/
# the code returns the fields interpolated to pressure surfaces so they can be used to compute PGF as gradfiuent if Z in the horizontal direction on constant pressure surfaces

# def main():
	# parser = argparse.ArgumentParser(prog='PyCLES')
	# parser.add_argument("wrf_path")
	# parser.add_argument("center_path")
	# args = parser.parse_args()
	# wrf_path = args.wrf_path
	# center_path = args.center_path

def WRF_fields(wrf_path, center_path):

	data = nc.Dataset(wrf_path, 'r')
	LON = np.squeeze(data.variables['XLONG'])
	LAT = np.squeeze(data.variables['XLAT'])
	day =   np.float(wrf_path[-14:-12])
	year =  np.float(wrf_path[-22:-18])
	month = np.float(wrf_path[-17:-15])
	hour =  np.float(wrf_path[-11:-9])

	# center = nc.Dataset(center_path, 'r')
	# X_center = np.array(center.variables['x_center'])
	# Y_center = np.array(center.variables['y_center'])
	# hour_center = np.array(center.variables['hour_center'])
	# month_center = np.array(center.variables['month_center'])
	# day_center = np.array(center.variables['day_center'])

	# current_index = np.where((day_center==day) & (hour_center==hour) & (month_center==month))[0][0]
	# x_center = -X_center[current_index]
	# y_center =  Y_center[current_index]

	P = np.squeeze(np.add(data.variables['PB'], data.variables['P']))
	P00 = np.array(data.variables['P00'])
	P_new = np.linspace(85000.0, 2000.0, num = 100)

	# convert the Lat Lon coordinates to X,Y [km] respectively to storm center location
	ind = np.unravel_index(np.argmin(P[2,:,:], axis=None), P[1,:,:].shape)
	x_center = LON[ind]
	y_center = LAT[ind]
	p_center = P[2,ind[0],ind[1]]
	y = np.multiply(111.1, np.subtract(LAT, y_center))
	x = np.multiply(111.1, np.multiply(np.subtract(LON, x_center), np.cos(np.deg2rad(LAT))))

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
	return 0.0, 0.0, x, y, P_out, Z, T

if __name__ == '__main__':
    main()