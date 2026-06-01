import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


iris = sns.load_dataset("iris")
print(iris.head())

iris_two = iris[(iris["species"] == "setosa") | (iris["species"] == "versicolor")]
x = iris_two.iloc[:, 0:2].to_numpy()
species = iris_two["species"]

y = species.map({"setosa": 0, "versicolor": 1}).to_numpy()

model = RandomForestClassifier(n_estimators=100, random_state=1)

model.fit(x, y)

print("Точность на этих же данных:", model.score(x, y))

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 100),
    np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 100),
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()

ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5])

plt.scatter(x[species == "setosa", 0], x[species == "setosa", 1], color="red", alpha=0.7, label="setosa")
plt.scatter(x[species == "versicolor", 0], x[species == "versicolor", 1], color="green", alpha=0.7, label="versicolor")

plt.title("RandomForestClassifier для двух сортов iris")
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.legend()
plt.grid(True)

plt.show()