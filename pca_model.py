import os
import numpy as np
from sklearn.linear_model import Ridge
import pylab as plt

from scipy.interpolate import RegularGridInterpolator


class PCAModel:

    def __init__(self, pca, mean, x_grid, y_grid, z_grid, scale, alpha=0.01,
                 num_basis_f=None):
        self.pca = pca
        self.mean = mean
        grid_shape = (len(x_grid), len(y_grid), len(z_grid))
        if num_basis_f is None:
            self.num_basis_fs = pca.shape[0]
        else:
            self.num_basis_fs = num_basis_f
        self.basis_f = []
        for i in range(self.num_basis_fs):
            pca3d = pca[i,:].reshape(grid_shape)
            self.basis_f.append(RegularGridInterpolator(
                (x_grid, y_grid, z_grid), pca3d))
        self.mean3d = RegularGridInterpolator(
            (x_grid, y_grid, z_grid), mean.reshape(grid_shape))
        self.scale = scale
        self.model = Ridge(alpha=alpha, fit_intercept=False)

    def fit(self, xs, ys, zs, vals):
        X = []  # shape is (num_points, num_basis_fs)
        Y = []  # shape is (num_points)
        for x, y, z, val in zip(xs, ys, zs, vals):
            X_row = []
            for i in range(self.num_basis_fs):
                X_row.append(self.basis_f[i]((x, y, z)).item() * self.scale[i])
            X.append(X_row)
            Y.append(val - self.mean3d((x, y, z)).item())

        self.model.fit(X, Y)

    def predict(self, xs, ys, zs):
        vals = []
        for x, y, z in zip(xs, ys, zs):
            val = self.mean3d((x, y, z)).item()
            for i in range(self.num_basis_fs):
                val += self.basis_f[i]((x, y, z)).item() * self.model.coef_[i] * self.scale[i]
            vals.append(val)

        return np.array(vals)

    def dummy_predict(self, xs, ys, zs):
        vals = []
        for x, y, z in zip(xs, ys, zs):
            val = self.mean3d((x, y, z)).item()
            vals.append(val)

        return np.array(vals)


calibration = np.load("pca_calibration_Z.npy", allow_pickle=True).item()
data = np.load("./train/0.npy", allow_pickle=True).item()

model = PCAModel(calibration["pca"], calibration["mean"],
                 data["X"][0,:], data["Y"][:,0], range(76), calibration["scale"],
                 alpha=0.01, num_basis_f=5)
model.fit(data["xs_flight"][70::76],
          data["ys_flight"][70::76],
          data["zs_flight"][70::76],
          data["Zs_flight"][70::76])

model.fit(data["X"].flatten(), data["Y"].flatten(), [level]*101*101, data["Z"][:,:,level].flatten())


model.fit(data["xs_flight"],
          data["ys_flight"],
          data["zs_flight"],
          data["Zs_flight"])
level = 70
ground_truth = data["Z"][:,:,level]
pred_zs = model.predict(data["X"].flatten(), data["Y"].flatten(), [level]*101*101)
dummy_pred_zs = model.dummy_predict(data["X"].flatten(), data["Y"].flatten(), [level]*101*101)

pred_zs = pred_zs.reshape((101, 101))
dummy_pred_zs = dummy_pred_zs.reshape((101, 101))
plt.subplot(2,1,1)
plt.contourf(pred_zs - ground_truth)
plt.colorbar()
plt.subplot(2,1,2)
plt.contourf(dummy_pred_zs - ground_truth)
plt.colorbar()
plt.show()
