from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(data)
PCA(n_components=2)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
first_pc = pca.components_[0]
second_pc = pca.components_[1]
print(first_pc)
print(second_pc)
transformed_data = pca.transform(data)    
for ii, jj in zip(transformed_data, data):
    plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], color="r")
    plt.scatter(second_pc[0]*ii[1], second_pc[1]*ii[1], color="c")
    plt.scatter(jj[0], jj[1], color="b")
plt.xlabel("bonous")
plt.ylabel("long-term incentive")
plt.show()

