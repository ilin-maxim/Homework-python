import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import viridis

# def f(x,y):
#     return np.sin(x * np.pi / 2 + np.sqrt(x**2 + y**2))

# x = np.linspace(-6, 6, 30)
# y = np.linspace(-10, 10, 50)
#
# X, Y = np.meshgrid(x, y)
# print(X.shape)
# print(Y.shape)
#
# Z = f(X, Y)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# # ax.scatter3D(X, Y, Z, c=Z)
# # ax.plot_wireframe(X, Y, Z)
# ax.plot_surface(X, Y, Z, cmap='binary')

# angle = np.linspace(0, 2 * np.pi, 50)
# r = np.linspace(0,6, 30)
# R, Angle = np.meshgrid(r, angle)
# X = R * np.sin(Angle)
# Y = R * np.cos(Angle)
# Z = f(X, Y)
# ax.plot_surface(X, Y, Z, cmap='viridis')

# angle = 1.5 * np.pi * np.random.random(1000)
# r = np.linspace(0,6, 1000)
# R, Angle = np.meshgrid(r, angle)
# X = r * np.sin(Angle)
# Y = r * np.cos(Angle)
# Z = f(X, Y)
# ax.plot_surface(X, Y, Z, cmap='viridis')
# plt.show()

# Seaborn

sns.set_style('darkgrid')
cars = pd.read_csv('cars.csv')
print(cars.head())

## Числовые данные
## Парная

# sns.pairlot(cars)
# sns.pairlot(data=cars, hue='transmission')

## Тепловая карта
# cars_coll=cars[['year', 'selling_price', 'seats', 'mileage']]
# sns.heatmap(cars_coll.corr(), cmap='viridis', annot=True)

## Диаграмма рассеивания
# sns.scatterplot(x='seats', y='mileage', data=cars, hue='fuel')
# sns.scatterplot(x='year', y='selling_price', data=cars)

## Диаграмма рассеивания + линейная регрессия
# sns.regplot(x='seats', y='mileage', data=cars)
# sns.relplot(x='seats', y='mileage', data=cars, kind='scatter', hue='transmission', col='fuel', col_wrap=2)
# sns.relplot(x='seats', y='mileage', data=cars, kind='line', hue='transmission', col='fuel', col_wrap=2)

# sns.lmplot(x='seats', y='mileage', data=cars, hue='transmission', col='fuel', col_wrap=2)

# Линейный график
# sns.lineplot(x='seats', y='mileage', data=cars, hue='fuel')

# Сводные диаграммы
# sns.jointplot(x='year', y='selling_price', data=cars, kind='kde')
# sns.jointplot(x='year', y='selling_price', data=cars, kind='hex')
# sns.jointplot(x='year', y='selling_price', data=cars, hue='transmission')

## Категории и числа
# sns.barplot(x='fuel', y='selling_price', data=cars, estimator=np.mean)
# sns.barplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission', col='seller_type')
# sns.catplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission', kind='bar', col='seller_type')
# sns.pointplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission', kind='bar', col='seller_type')

# sns.boxplot(x='fuel', y='selling_price', data=cars, hue='transmission')
# sns.violinplot(x='fuel', y='selling_price', data=cars, hue='transmission')

# sns.stripplot(x='fuel', y='selling_price', data=cars, hue='transmission')

plt.show()





