# SciKit - Learn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


iris = sns.load_dataset("iris")
print(iris.head())

# print(type(iris))
# print(type(iris.values))
# print(type(iris.values.shape))
# print(type(iris.columns))
# print(type(iris.index))
#
# sns.pairplot(iris, hue="species")

# plt.show()

# Строки - образцы - отдельный объект (sample)
# Столбцы - признаки (feature)
# Матрица признаков [число образцов на число признаков] - признаки - независимая переменная
# Целевой массив (target, label) [1 на число образцов] - зависимая переменная

x_iris = iris.drop("species", axis=1)
# print(x_iris)
#
y_iris = iris['species']
# print(y_iris)

# 1. Выбирается класс модели
# 2. Выбираются гиперпараметры модели
# 3. На основе данных создаётся матрица признаков и целевой вектор
# 4. Обучение модели fit()
# 5. Обученная модель применяется к новым данным
#   5.1 Обучение с учителем - predict()
#   5.2 Обучение без учителя - predict() или transform()

# С учителем. Регрессия. Линейная регрессия
x = iris[iris['species'] == 'setosa'].iloc[:, 0].to_numpy()
y = iris[iris['species'] == 'setosa'].iloc[:, 1].to_numpy()

# 1. Выбирается класс модели
from sklearn.linear_model import LinearRegression

# 2. Выбираются гиперпараметры модели
model = LinearRegression(fit_intercept=False)

# 3. На основе данных создаётся матрица признаков и целевой вектор

# 4. Обучение модели fit()
reg = model.fit(x[:, np.newaxis], y)

plt.scatter(x, y)
# 5. Обученная модель применяется к новым данным
#   5.1 Обучение с учителем - predict()

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict(xfit[:, None])

plt.plot(xfit, yfit, "r")
plt.plot(xfit, xfit * reg.coef_ + reg.intercept_, "k")
# y = kx + b

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(2), LinearRegression())
reg = model.fit(x[:, np.newaxis], y)

# xfit = np.linspace(x.min(), x.max(), 1000)
# yfit = model.predict(xfit[:, None])
#
# plt.plot(xfit, yfit, "r")
#
# plt.show()


# Классификация. Логистическая регрессия

x_0 = iris[iris['species'] == 'setosa'].iloc[:, 0].to_numpy()
y_0 = iris[iris['species'] == 'setosa'].iloc[:, 1].to_numpy()
x_1 = iris[iris['species'] == 'versicolor'].iloc[:, 0].to_numpy()
y_1 = iris[iris['species'] == 'versicolor'].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

x_00 = iris[iris['species'] == 'setosa'].iloc[:, 0].to_numpy()
x_11 = iris[iris['species'] == 'versicolor'].iloc[:, 0].to_numpy()

plt.scatter(x_00, np.full(50, 1), color="red", alpha=0.5)
plt.scatter(x_11, np.full(50, 1), color="green", alpha=0.5)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

x = iris[iris['species'] != 'virginica'].iloc[:, 0].to_numpy()
y = iris[iris['species'] != 'virginica'].iloc[:, 4]

model.fit(x[:, None], y)

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict_proba(xfit[:, None])

# plt.plot(xfit, 1 + 4 * yfit[:, 1], "green")
# plt.plot(xfit, 1 + 4 * yfit[:, 0], "red")
# plt.show()

# Деревья решений
from sklearn.tree import DecisionTreeClassifier

x = iris[iris['species'] != 'virginica'].iloc[:, 0:2].to_numpy()
y = iris[iris['species'] != 'virginica'].iloc[:, 4]
y1 = np.full(50, 1)
y2 = np.full(50, 2)
y = np.ravel([y1, y2])

tree = DecisionTreeClassifier()
tree.fit(x, y)

print(np.c_[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
print(np.ravel([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]))


xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax = plt.gca()
ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

# plt.show()

# Метод опорных векторов. Классификация

from sklearn.svm import SVC
model = SVC(kernel="linear", C=1e10)

x = iris[iris['species'] != 'virginica'].iloc[:, 0:2].to_numpy()
y = iris[iris['species'] != 'virginica'].iloc[:, 4]

model.fit(x, y)

ax = plt.gca()
x_0 = iris[iris['species'] == 'setosa'].iloc[:, 0].to_numpy()
y_0 = iris[iris['species'] == 'setosa'].iloc[:, 1].to_numpy()
x_1 = iris[iris['species'] == 'versicolor'].iloc[:, 0].to_numpy()
y_1 = iris[iris['species'] == 'versicolor'].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

plt.show()

# Наивная базисовская классификация
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()
ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

x_m = model.theta_[0]
y_m = model.theta_[1]
x_var = model.var_[0]
y_var = model.var_[1]

z1 = 1 / (2 * np.pi * (x_var[0] * x_var[1]) ** 0.5) * np.exp(- ((xx - x_m[0]) ** 2) / (2 * x_var[0]) - ((yy - x_m[1]) ** 2) / (2 * x_var[1]))
z2 = 1 / (2 * np.pi * (y_var[0] * y_var[1]) ** 0.5) * np.exp(- ((xx - y_m[0]) ** 2) / (2 * y_var[0]) - ((yy - y_m[1]) ** 2) / (2 * y_var[1]))



# ax.contour(xx, yy, z1)
# ax.contour(xx, yy, z2)
#
# ax = plt.axes(projection='3d')
# ax.contour3D(xx, yy, z1, 50)
# ax.contour3D(xx, yy, z2, 50)
#
# plt.show()

# к - ближ. соседей
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x, y)
xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

