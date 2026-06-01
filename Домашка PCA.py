import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


iris = sns.load_dataset("iris")
print(iris.head())

iris_two = iris[(iris["species"] == "setosa") | (iris["species"] == "versicolor")]
x = iris_two.drop("species", axis=1).to_numpy()
y = iris_two["species"]

model = PCA(n_components=2)

x_pca = model.fit_transform(x)

setosa = y == "setosa"
versicolor = y == "versicolor"

plt.scatter(x_pca[setosa, 0], x_pca[setosa, 1], color="red", alpha=0.5, label="setosa")
plt.scatter(x_pca[versicolor, 0], x_pca[versicolor, 1], color="green", alpha=0.5, label="versicolor")

plt.title("PCA для двух сортов iris")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.legend()
plt.grid(True)

plt.show()