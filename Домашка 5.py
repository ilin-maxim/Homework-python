import numpy as np
import pandas as pd
# Pandas
# Series
# DataFrame
# Index

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print(data.values)
print(type(data.values))

print(data.index)
print(type(data.index))

print(data[1])
print(data[1:3])

data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
print(data)
print(type(data.index[0]))
print(data["a"])
print(data["b":"d"])


data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1, 10, 7, 'd'])
print(data)
print(type(data.index))
print(type(data.index[0]))
print(type(data.index[3]))

print(data[1])
print(data[10:"d"])
print(data["d"])

dict = {
    "a": 10,
    "b": 20,
    "c": 30,
    "d": 40,
    "e": 50,
}
data_dict = pd.Series(dict)
print(data_dict)
print(data_dict["b"])
print(data_dict["b":"d"])

# Списки
data = pd.Series(5, index=[20, 30, 40])
print(data)
# Словари
data_dict = pd.Series(dict, index=["a", "d"])
print(data_dict)

# DataFrame
dict1 = {
    "a": 10,
    "b": 20,
    "c": 30,
    "d": 40,
    "e": 50,
    "f": 60,
}
dict2 = {
    "a": 11,
    "b": 22,
    "c": 33,
    "d": 44,
    "e": 55,
    "h": 77
}
data_dict1 = pd.Series(dict1)
data_dict2 = pd.Series(dict2)

df = pd.DataFrame({"Dict_1": data_dict1, "Dict_2": data_dict2})
print(df)

print(df.values)
print(type(df.values))

print(df.columns)
print(type(df.columns))
print(df.index)
print(type(df.index))

print(df["Dict_1"])
print(df["Dict_2"])

print(type(df["Dict_2"]))

df = pd.DataFrame({"Dict_1": data_dict1})
print(df)
print(df.values)
df = pd.DataFrame(data_dict1 , columns=["rrr"])
print(df)
print(df.values)

print(pd.DataFrame([{'a': i, 'b': 2*i} for i in range(10)]))
# Nan - Not a Number

print(pd.DataFrame(np.zeros(3)))

# Index - способ сослаться (reference) на данные в Series или в DataFrame
index = pd.Index([2, 5, 3, 71])
print(index)
print(type(index))

print(index[0])
print(index[::2])

index1 = pd.Index([2, 5, 3, 71])
index2 = pd.Index([1, 2, 5, 87, 3, 71])
print(index1.intersection(index2))
print(index1.union(index2))

data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
print(data['a'])
# print('a' in data)
# print('a1' in data)
# print(list(data.items()))
# data['a'] = 99
# print(data)
#
# data['a1'] = 990
# print(data)
#
# print(data['c':'a1'])
# print(data[2:4])
# print(data[2:])


print(data[(data > 0.3) & (data < 1)])
print(data.loc[['c', 'a']])
print(data.iloc[[2, 0]])

# Атрибуты-индексаторы


dict1 = {
    "a": 10,
    "b": 20,
    "c": 30,
    "d": 40,
    "e": 50,
    "f": 60,
}
dict2 = {
    "a": 11,
    "b": 22,
    "c": 33,
    "d": 44,
    "e": 55,
    "h": 77
}
data_dict1 = pd.Series(dict1)
data_dict2 = pd.Series(dict2)

df = pd.DataFrame({"Dict_1": data_dict1, "Dict_2": data_dict2})
# print(df['Dict_1'])
# print(df.Dict_1)

df['new'] = df['Dict_1']
print(df)
df['new1 '] = df['Dict_2'] / df['Dict_1']
print(df)

print(df.values)
print(df.T)
print(df['Dict_1']) # Столбец
print(df.values[0])

print(df.loc['a':  'c', :'Dict_2'])
print(df.iloc[:3, :2])
# print(df.loc[df.Dict_2 > 30, ['new1', 'Dict_1']])
df.loc[df.Dict_2 > 30, ['new1', 'Dict_1']] = 36
print(df)

rng = np.random.default_rng(1)
s = pd.Series(rng.integers(0, 10, 6))
print(s)
print(np.exp(s))

df1 = pd.DataFrame(rng.integers(0, 10, (3, 4)), columns=['a', 'b', 'c', 'd'])
print(df1)

print(np.sin(df1 * 4))

print(data_dict1 + data_dict2)
print(data_dict1.add(data_dict2, fill_value=5))
print(df1 + df)
print(df.add(df1, fill_value=df.values.sum()))

A = rng.integers(0, 10, (3, 4))
print(A[0])
print(A - A[0])

df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
print(df)
print(df -  df.iloc[0])
print(df.subtract(df['a'], axis=0))

