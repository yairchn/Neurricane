import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
from generate_TC_fields import TC_fields
from WRF_sampler import WRF_fields, WRF_channel_fields

# command line:
# python flight_sampler.py    resolution    oriantation1  oriantation2       RMW      width_ratio     east_offset  north_offset       mode
# python flight_sampler.py    1.0             0.1            0.5              0.3          0.9            0.9             0.9        wrf / wrf_channel/ dummy

# The model recieves 7 input variables thbat has values between 0 and 1
# resolution is the spacing of the sampling (dropsondes). The code takes a value between 0 and 1 and convert it to a range [3, 10].
# oriantation1, oriantation2 are values between 0 and 1 that determine the flight angle. If they are [1,0] the flight paths will be prependicular. if the are idnetical the flight paths will in be 45deg
# RMW is the Radius of Maximum wind [m] at the 850mb level. The code takes a value between 0 and 1 and convert it to a range [30, 100]km
# width ratio is the ratio of widths of the warm core and the surface low (RMW). The code takes a value between 0 and 1 and convert it to a range [0.0, 2.0]
# offsets are the dispacments on the x asnd y axes of the warm core with respect to the surface low. They are given in units of RMW, should be between 0 and 1, 0
# the outputs from this file are:
# xc,yc, the location ofthe storm center; Z, T the 3D temperature and geopotential fields and their cartesian coordinates X[:,:], Y[:,:], p[:]
# {Z,T,Z,Y}_flight# are the data sampled along flight path numbered #

def main():
	parser = argparse.ArgumentParser(prog='PyCLES')
	parser.add_argument("resolution", type=float)
	parser.add_argument("oriantation1", type=float)
	parser.add_argument("oriantation2", type=float)
	parser.add_argument("RMW", type=float)
	parser.add_argument("width_ratio", type=float)
	parser.add_argument("east_offset", type=float)
	parser.add_argument("north_offset", type=float)
	parser.add_argument("mode", type=str)

	args = parser.parse_args()
	resolution = args.resolution
	oriantation1 = args.oriantation1
	oriantation2 = args.oriantation2
	RMW = args.RMW
	width_ratio = args.width_ratio
	east_offset = args.east_offset
	north_offset = args.north_offset
	oriantation2 = 1.0-oriantation2
	mode = args.mode

	# take the random values between 0 and 1 and convert to relevent sizes
	# resolution = 1.0+ round(resolution*50.0)
	RMW = 30000 + RMW*100000 # between 30km and 130km
	width_ratio = width_ratio*2.0
	east_offset  = (east_offset-1.0)/2.0*1.5*RMW # offset if proportional to storm size
	north_offset = (north_offset-1.0)/2.0*1.5*RMW # offset if proportional to storm size
	resolution = 3.0 + resolution*7.0

	if mode == 'dummy':
		xc, yc, X, Y, p, Z, T, i850, i150 = TC_fields(RMW ,width_ratio,east_offset, north_offset)
	elif mode == 'wrf':
		wrf_path = '/Volumes/TimeMachine/hurricanes/Patricia_for_Yair/wrfout_d04_2015-10-23_04_00_00.nc'
		xc, yc, X, Y, p, Z, T, i850, i150 = WRF_fields(wrf_path)
	elif mode == 'wrf_channel':
		wrf_path = '/Users/yaircohen/Documents/DanFu/ENP2017d02_Rog_HUE09_0015_-3.00.nc'
		xc, yc, X, Y, p, Z, T, i850, i150 = WRF_channel_fields(wrf_path)
	else:
		print('mode is not recognized')

	I,J,K = np.shape(Z)
	num1 = np.round(I/resolution-1)
	num2 = np.round(J/resolution-1)

	i1 = np.round(np.linspace(0,I-1,num1))
	j1 = np.round(i1*oriantation1)
	j1 += np.round(I/2)-j1[int(np.round(len(j1)/2))]

	j2 = i1
	i2 = np.round(oriantation2*j2)
	i2 = np.max(i2)-i2
	i2 += np.round(J/2)-i2[int(np.round(len(i2)/2))]

	x1 = [int(i) for i in i1]
	y1 = [int(i) for i in j1]
	x2 = [int(i) for i in i2]
	y2 = [int(i) for i in j2]

	Z_flight1 = Z[x1,y1]
	T_flight1 = T[x1,y1]
	X_flight1 = X[x1,y1]
	Y_flight1 = Y[x1,y1]
	Z_flight2 = Z[x2,y2]
	T_flight2 = T[x2,y2]
	X_flight2 = X[x2,y2]
	Y_flight2 = Y[x2,y2]

	plt.figure('Z flight1')
	plt.plot(Z_flight1)
	plt.figure('Z flight2')
	plt.plot(Z_flight2)

	plt.figure('T flight1')
	plt.plot(T_flight1)
	plt.figure('T flight2')
	plt.plot(T_flight2)

	plt.figure('upper level Z')
	plt.contour(X, Y, Z[:,:,i150])
	plt.plot(i1,j1,'.r')
	plt.plot(i2,j2,'.b')
	plt.figure('lower level Z and mid level T')
	plt.contour(X, Y, T[:,:,int((i150-i150%2)/2)])
	plt.contour(X, Y, Z[:,:,i850])
	plt.plot(i1,j1,'.r')
	plt.plot(i2,j2,'.b')
	plt.show()

	return xc, yc, X, Y, Z, T, Z_flight1, Z_flight2, T_flight1, T_flight2, X_flight1, X_flight2, Y_flight1, Y_flight2


if __name__ == '__main__':
    main()