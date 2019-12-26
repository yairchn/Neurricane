import random
import os
import numpy as np
from generate_TC_fields import TC_fields


def gen_sample(resolution, oriantation1, oriantation2, RMW, width_ratio, east_offset, north_offset):
    # take the random values between 0 and 1 and convert to relevent sizes
    # resolution = 1.0+ round(resolution*50.0)
    RMW = 30000 + RMW*100000 # between 30km and 130km
    width_ratio = width_ratio*2.0
    east_offset  = (east_offset-1.0)/2.0*1.5*RMW # offset if proportional to storm size
    north_offset = (north_offset-1.0)/2.0*1.5*RMW # offset if proportional to storm size
    resolution = 3.0 + resolution*7.0

    xc, yc, X, Y, p, Z, T = TC_fields(RMW ,width_ratio,east_offset, north_offset)
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

    x1 = np.array([int(i) for i in i1])
    y1 = np.array([int(i) for i in j1])
    x2 = np.array([int(i) for i in i2])
    y2 = np.array([int(i) for i in j2])
    Z_flight1 = Z[x1,y1]
    T_flight1 = T[x1,y1]
    X_flight1 = X[x1,y1]
    Y_flight1 = Y[x1,y1]
    Z_flight2 = Z[x2,y2]
    T_flight2 = T[x2,y2]
    X_flight2 = X[x2,y2]
    Y_flight2 = Y[x2,y2]

    xs_flight = []
    ys_flight = []
    zs_flight = [] # xs, ys, zs are coordinates
    Ts_flight = []
    Zs_flight = [] # Ts, Tz are measurements
    zs_dim = Z.shape[2]
    for i in range(len(X_flight1)):
        for j in range(zs_dim):
            xs_flight.append(X_flight1[i])
            ys_flight.append(Y_flight1[i])
            zs_flight.append(j)
            Ts_flight.append(T_flight1[i, j])
            Zs_flight.append(Z_flight1[i, j])

    for i in range(len(X_flight2)):
        for j in range(zs_dim):
            xs_flight.append(X_flight2[i])
            ys_flight.append(Y_flight2[i])
            zs_flight.append(j)
            Ts_flight.append(T_flight2[i, j])
            Zs_flight.append(Z_flight2[i, j])

    data = {
        "X": X,
        "Y": Y,
        "p": p,
        "Z": Z,
        "T": T,
        "xs_flight": np.array(xs_flight),
        "ys_flight": np.array(ys_flight),
        "zs_flight": np.array(zs_flight),
        "Ts_flight": np.array(Ts_flight),
        "Zs_flight": np.array(Zs_flight)
    }

    # adding some noise to observations
    NOISE_LEVEL = 0.0
    for var_name in ["Ts_flight", "Zs_flight"]:
        data[var_name] *= np.random.uniform(1-NOISE_LEVEL, 1+NOISE_LEVEL, data[var_name].shape)
    return data


N = 10
save_dir = "test"
LOWER_BOUND = 0.2
UPPER_BOUND = 0.8

for i in range(N):
    resolution = random.uniform(LOWER_BOUND, UPPER_BOUND)
    oriantation1 = random.uniform(LOWER_BOUND, UPPER_BOUND)
    oriantation2 = random.uniform(LOWER_BOUND, UPPER_BOUND)
    RMW = random.uniform(LOWER_BOUND, UPPER_BOUND)
    width_ratio = random.uniform(LOWER_BOUND, UPPER_BOUND)
    east_offset = random.uniform(LOWER_BOUND, UPPER_BOUND)
    north_offset = random.uniform(LOWER_BOUND, UPPER_BOUND)
    data = gen_sample(resolution, oriantation1, oriantation2, RMW, width_ratio, east_offset, north_offset)
    save_file_path = os.path.join(save_dir, "%d.npy" % i)
    np.save(save_file_path, data)