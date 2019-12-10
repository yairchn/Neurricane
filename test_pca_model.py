import os
import random

import numpy as np
import pylab as plt
from pca_model import PCAModel


def cal_test_err(calibration, data):
    model = PCAModel(calibration["pca"], calibration["mean"],
                     data["X"][0, :], data["Y"][:, 0], range(76), calibration["scale"],
                     alpha=0.01, num_basis_f=15)
    model.fit(data["xs_flight"], data["ys_flight"], data["zs_flight"], data["Zs_flight"])

    ground_truth = data["Z"]

    # sample random 100 points for testing
    N = 100
    xs = np.zeros(N)
    ys = np.zeros(N)
    zs = np.zeros(N)
    vals = np.zeros(N)
    grid_shape = (101, 101, 76)
    for i_sample in range(N):
        i = random.randint(0, grid_shape[0] - 1)
        j = random.randint(0, grid_shape[1] - 1)
        k = random.randint(0, grid_shape[2] - 1)
        xs[i_sample] = data["X"][i, j]
        ys[i_sample] = data["Y"][i, j]
        zs[i_sample] = k
        vals[i_sample] = ground_truth[i,j,k]

    pca_predictions = model.predict(xs, ys, zs)
    dummy_predictions = model.dummy_predict(xs, ys, zs)

    pca_predictions_err = np.mean((vals - pca_predictions)**2) ** 0.5
    dummy_predictions_err = np.mean((vals - dummy_predictions)**2) ** 0.5
    return pca_predictions_err, dummy_predictions_err


calibration = np.load("pca_calibration_Z.npy", allow_pickle=True).item()
data_dir = "test"
data_files = os.listdir(data_dir)

pca_errs = []
dummy_errs = []

for i in range(len(data_files)):
    data = np.load(os.path.join(data_dir, data_files[i]), allow_pickle=True).item()
    pca_err, dummy_err = cal_test_err(calibration, data)
    pca_errs.append(pca_err)
    dummy_errs.append(dummy_err)

plt.plot(range(len(data_files)), pca_errs, 'o', label='pca model error', markersize=2)
plt.plot(range(len(data_files)), dummy_errs, 'o', label='dummy model error', markersize=2)
plt.legend()
plt.savefig("pca_vs_dummy_errors.png")
plt.show()

plt.figure()
plt.scatter(pca_errs, dummy_errs, s=1)
plt.xlabel('pca')
plt.show()