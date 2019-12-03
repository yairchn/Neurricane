import numpy as np
import netCDF4 as nc
import pylab as plt

# command line:
# python generate_TC_fields.py RMW    width_ratio  x_offset  y_offset
# python generate_TC_fields.py 50000  1.2          30000      30000
def TC_fields(RMW ,width_ratio ,x_offset, y_offset):

	Rd = 287.4
	g = 9.81
	f = 5*1e-5
	cp = 1004.0

	# generate cartisian fields
	dx = 1000.0
	nx = 100000.0/dx+1
	x = np.linspace(-600000.0, 600000.0, nx)
	y = np.linspace(-600000.0, 600000.0, nx)
	X, Y = np.meshgrid(x, y)
	p = np.linspace(850.0,100.0,76)
	Z0 = 1300
	Za = -300
	WZ = RMW
	Zb = Z0 + np.multiply(Za,np.exp(- np.divide(np.add(np.multiply(X,X),np.multiply(Y,Y)),(2.0*WZ*WZ))))


	# generate a realistic mean temperature profile for the tropics following Dunion (2011).
	T0 = np.zeros_like(p)
	T0[0]  =  17.1 + 273.14
	T0[15] =   9.2 + 273.14
	T0[45] = -17.6 + 273.14
	T0[55] = -33.4 + 273.14
	T0[65] = -54.9 + 273.14

	T0 [1:15]  = T0[0]+ (T0[15]-T0[0]) /(p[15]- p[0]) * (p[1:15]- p[0] )
	T0 [16:45] = T0[15]+(T0[45]-T0[15])/(p[45]-p[15]) * (p[16:45]-p[15])
	T0 [46:55] = T0[45]+(T0[55]-T0[45])/(p[55]-p[45]) * (p[46:55]-p[45])
	T0 [56:65] = T0[55]+(T0[65]-T0[55])/(p[65]-p[55]) * (p[56:65]-p[55])
	T0 [66:76] = T0[65]+(T0[65]-T0[55])/(p[65]-p[55]) * (p[66:76]-p[65])

	# vertical structure of the Temperature anomaly (Ta) as a Gaussian with maximum at 400hPa
	Ta = np.multiply(10.0,np.exp(-np.divide( np.power(np.subtract(p,400),2),np.multiply(2.0,110*110))))

	WT = WZ*width_ratio
	X0 = x_offset
	Y0 = y_offset
	xc,yc = np.argwhere(Zb == np.min(Zb))[0]
	x1,y1 = np.shape(Zb)

	Z = np.zeros([len(x),len(y),len(p)])
	T = np.zeros([len(x),len(y),len(p)])
	Z[:,:,0] = Zb
	T[:,:,0] = np.add(T0[0], np.multiply(Ta[0],  np.exp(-np.divide( (np.add(np.power(np.subtract(X,X0),2.0),np.power(np.subtract(Y,Y0),2))),(2.0*WT*WT)))))

	for i in range(len(p)):
		T[:,:,i] = np.add(T0[i], np.multiply(Ta[i],  np.exp(-np.divide( (np.add(np.power(np.subtract(X,X0),2.0),np.power(np.subtract(Y,Y0),2))),(2.0*WT*WT ))) ))

	Z[:,:,0] = Zb
	for i in range(1,len(p)):
	    Z[:,:,i] = np.add(Z[:,:,i-1], np.multiply(Rd*np.log(p[i-1]/p[i])/g/2,np.add(T[:,:,i-1],T[:,:,i])))

	return xc, yc, X, Y, p, Z, T

if __name__ == '__main__':
    main()