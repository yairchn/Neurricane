import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
from generate_TC_fields import TC_fields

# command line:
# python flight_sampler.py resolution    oriantation1  oriantation2  RMW    width_ratio  offset
# python flight_sampler.py 10             0.1            0.5         50000  1.2          0.7

# Here oriantation1 and 2 are values between 0 and 1 that determine the flight angle. if they are identical to eachother the lines will be prependicular
# RMW is the Radius of Maximum wind [m] at the 850mb level should range between 30 and 100km.
# width ratio is the ratio of width of the warm core and the surface low (RMW). it should be between 0.5 and 2.0
# off set if the dispacment on the x axis of the warm core with respect to the surface low. It is given in units of RMW, should be between 0 and 1
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
	parser.add_argument("offset", type=float)

	args = parser.parse_args()
	resolution = args.resolution
	oriantation1 = args.oriantation1
	oriantation2 = args.oriantation2
	RMW = args.RMW
	width_ratio = args.width_ratio
	offset_ = args.offset
	offset = offset_*RMW

	xc, yc, X, Y, p, Z, T = TC_fields(RMW ,width_ratio,offset)
	I,J,K = np.shape(Z)
	i1 = np.linspace(0,I-1,I/resolution)
	j1 = np.round(oriantation1*i1)

	j1 += xc-j1[int(np.round(I/2/resolution))]

	j2 = np.linspace(0,J-1,J/resolution-1)
	i2 = np.round(oriantation2*j2)

	i2 += xc-i2[int(np.round(I/2/resolution))]
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

	plt.figure('flight1')
	plt.plot(flight1)
	plt.figure('flight2')
	plt.plot(flight2)

	plt.figure('upper level Z')
	plt.contour(Z[:,:,75])
	plt.plot(i1,j1,'.')
	plt.plot(i2,j2,'.')
	plt.figure('lower level Z and mid level T')
	plt.contour(T[:,:,40])
	plt.contour(Z[:,:,0])
	plt.show()

	return xc, yc, X, Y, Z, T, Z_flight1, Z_flight2, T_flight1, T_flight2, X_flight1, X_flight2, Y_flight1, Y_flight2


if __name__ == '__main__':
    main()