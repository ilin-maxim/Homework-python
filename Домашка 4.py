import numpy as np
import matplotlib.pyplot as plt


# Простые индексы

x = np.arange(12).reshape(3, 4)
print(x)
print(x[2])
print(x[2, [2, 0, 1]])
print(x[[2, 0, 1], 2 ])

# Срезы

print(x[1:])
print(x[1:, [2, 0, 1]])

# Маски
mask = np.array([1, 1, 1, 0], dtype=bool)
print(mask)
print(mask.shape)
row = np.array([0, 2 ])
print(row[:, np.newaxis].shape )
print(x[row[:, np.newaxis], mask])

mask = np.array([1, 1, 1, 1], dtype=bool)
row = np.array([0, 1, 2 ])
print(x[row[:, np.newaxis], mask])



# rng = np.random.default_rng(seed=1)
# x = rng.multivariate_normal([0, 0], [[1, 2], [2, 5]], 100)
#
# np.random.seed(0)
# inx = np.random.choice(100, 30, replace=False)
# print(inx)
# select = x[inx]
# plt.scatter(x[:, 0], x[:, 1], alpha=.3)
# plt.scatter(select[:, 0], select[:, 1], s=200, facecolors='none', edgecolors='r')
#
# plt.show()

x = np.arange(10)
print(x)
inx = np.array([2, 8, 4, 1, 4 ])
# x[inx] = 99
# x[inx] += 1
np.add.at(x, inx, 1)
print(x)

rng = np.random.default_rng(seed=1)
x = rng.integers(100, size=100)
print(x[:10])

bins = np.linspace(0, 100, 11 )
counts = np.zeros(11)
i = np.searchsorted(bins, x)
np.add.at(counts, i, 1)
print(i[:10])
print(counts)
print(bins)
print(counts)

a = [1, 2, 3, 32, 3, 2, 34, 4, 44, 28, 18, 9, 54, 42, 0, 4, 49, 3, 86, 83, 86, 98, 53, 2, 4, 5, 99, 1, 435]
print(sorted(a))
a.sort()
print(a)

x = np.array(a)
print(x)
print(np.sort(x))
x.sort()
print(x)

inx = np.argsort(x)
print(inx)


rng = np.random.default_rng(seed=1)
x = rng.integers(0, 10, size=(4, 6))
print(x)
print(np.sort(x, axis=0))

# Структурированные массивы
# Массивы записей

name = ["Ирина", 'Виталий', "Олег", "Саша"]
age = [25, 17, 52, 44]
weight = [55.0, 57, 78, 72]
data = np.zeros(4, dtype={
    'names': ("name_", "age_", "weight_"),
     'formats': ("U10", "i4", "f8")
})

print(data.dtype)
data["name_"] = name
data["age_"] = age
data["weight_"] = weight
print(data)
print(data["name_"])


data_rec = data.view(np.recarray)
print(data_rec)
print(data_rec.name_)
print(data_rec.age_)
print(data_rec.weight_)

print(data["age_"] < 30)

# x["mat"][0] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# tp = np.dtype([("id", "i8"), ("mat", "f8", (3, 3))])
# x = np.zeros(2, dtype=tp)

data = [("Ирина", 25, 55), ("Виталий", 17, 57)]
dtype={
    'names': ("name_", "age_", "weight_"),
     'formats': ("U10", "i4", "f8")
}

data_rec = np.rec.array(data, dtype=dtype)
print(data_rec.name_)
