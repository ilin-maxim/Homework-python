import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


iris = sns.load_dataset("iris")
print(iris.head())

iris_two = iris[(iris["species"] == "setosa") | (iris["species"] == "versicolor")]
x = iris_two.iloc[:, 0:2].to_numpy()

model = KMeans(n_clusters=2, random_state=1, n_init=10)

clusters = model.fit_predict(x)

centers = model.cluster_centers_

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 100),
    np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 100),
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()

ax.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(x[:, 0], x[:, 1], c=clusters, cmap="viridis", alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], color="red", marker="X", s=200, label="Центры кластеров")

plt.title("KMeans для двух сортов iris")
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.legend()
plt.grid(True)

plt.show()