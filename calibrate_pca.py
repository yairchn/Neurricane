import os
import numpy as np
from sklearn.decomposition import PCA

data_dir = "train"
data_files = os.listdir(data_dir)

var_name = "Z"
var_composite = []
for f in data_files:
    data = np.load(os.path.join(data_dir, f), allow_pickle=True).item()
    var_composite.append(data[var_name].flatten())

pca_model = PCA(n_components=30)
pca_model.fit(var_composite)
coef = pca_model.transform(var_composite)
scale = np.std(coef, axis=0)

np.save("pca_calibration_%s.npy" % var_name,
        {"pca": pca_model.components_,
         "mean": pca_model.mean_,
         "scale": scale})