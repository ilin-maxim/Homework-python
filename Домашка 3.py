import numpy as np
import matplotlib.pyplot as plt

#Правила транслирования (broadcasting)

#1. Сравниваются размерности двух массивов. Если размерности отличаются, то форма массива с меньшей размерностью
# дополняется единицами с левой стороны

#2. Если формы двух массивов не совпадают в каком-то измерении, то массив с формой, равной единицы в данном измерении,
# растягивается до соответсвия форме другого массива

#3. Если в каком либо измерении размеры массивов различаются и ни один не равен единице, то генерируется ошибка

a = np.ones((2, 3))
b = np.arange(3)
c = np.ones((1, 3))

print(a)
print(b)
print(c)


print(a.shape)
print(b.shape)
print(c.shape)

c = a + b
print(c)
print(c.shape)

a = np.arange(3).reshape((3, 1))
b = np.arange(3)
print(a)
print(b)
print(a.shape)
print(b.shape)

print(a + b)


a = np.ones((3, 2))
b = np.arange(3)
print(a)
print(b)
print(a.shape)
print(b.shape)
#print(a + b)

# Центрирование массивов
a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1 ]])
print(a)

aMean = a.mean(0)
print(aMean)

print(a.shape)

aCentr = a - aMean
print(aCentr)

aMean = a.mean(1)
print(aMean)
print(aMean.shape)

aMean =  aMean[:, np.newaxis]
print(aMean)
print(aMean.shape)
aCentr = a - aMean
print(aCentr)
print(aCentr.mean(1))

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
print(x.shape)
print(y.shape)
y = y[:, np.newaxis]
# z = np.sin(x) * y + np.cos(10 + y * x) ** 3
# plt.imshow(z)
# plt.colorbar()
# plt.show()

#  Маскирование

x = np.arange(1, 6)
print(x)
print(x < 3)
print(np.less(x, 3))

rng1 = np.random.default_rng(seed=1)
rng2 = np.random.default_rng(seed=10)
x = rng1.integers(10, size=(3, 4))

print(x)
print(x < 6)

# Сколько элементов имеют значение меньше 6
print(np.count_nonzero(x < 6))
print(np.sum(x < 6))

print(np.sum(x < 6, axis=0))
print(np.sum(x < 6, axis=1))

print(np.any(x > 8))
print(np.all(x < 10))
print(np.all(x != 10))

print(np.sum((x > 3) & (x < 9), axis=0))
print(np.sum(np.bitwise_and(np.greater(x, 3), np.less(x, 9)), axis=0))

# Наложение маски
print(x < 5)
print(x[x<5])

print(bool(42), bool(0))
print(bool(42 and 0))
print(bool(42 or 0))

print(bin(42))
print(bin(59))
print(bin(42 & 59))
print(bin(42 | 59))

print(bin(42 and 59))
print(bin(42 or 59))

a = np.array([1, 0, 0, 1, 1, 0], dtype=bool)
b = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
print(a & b)
# print(a and b)

# Способы доступа к элементам массива
a = np.arange(10)
print(a)
print(a[3])
print(a[3:4])
print(a[a == 3])

# Векторизованная / прихотливая индексация
a = np.arange(1, 10)
ind = [3, 5, 0]
print(a[ind])

ind = [[3, 5], [5, 0]]
print(a[ind])


a = np.arange(12).reshape((3, 4))
print(a)
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
print(a[row, col])
print(a[row[:, np.newaxis], col])






