import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

iris = sns.load_dataset("iris")

print(iris.head())
species_int = []
for row in iris.values:
    match row[4]:
        case 'setosa':
            species_int.append(1)
        case 'versicolor':
            species_int.append(1)
        case 'virginica':
            species_int.append(1)

species_int_df = pd.DataFrame(species_int)
print(species_int_df.head())

data = iris[["sepal_length", "petal_length"]]
data['species'] = species_int

print(data.head())

# data_df = data[(data["species"] == 3) | (data["species"] == 2)]

data_of_virginica = data[data["species"] == 3]
data_of_versicolor = data[data["species"] == 2]
data_of_setosa = data[data["species"] == 1]

data_of_virginica_A = data_of_virginica.iloc[:25, :]
data_of_virginica_B = data_of_virginica.iloc[:25, :]

data_of_versicolor_A = data_of_versicolor.iloc[:25, :]
data_of_versicolor_B = data_of_versicolor.iloc[:25, :]

data_df_A = pd.concat([data_of_virginica_A, data_of_versicolor_A], ignore_index = True)
data_df_B = pd.concat([data_of_virginica_A, data_of_versicolor_B], ignore_index = True)



x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]))
x2_p = np.linspace(min(data_df["pepal_length"]), max(data_df["pepal_length"]))

x1_p, x2_p = np.meshgrid(x1_p, x2_p)
print(x1_p.shape)

X_p = pd.DataFrame(np.vstack([x1_p.ravel(), x2_p.ravel()]).T, columns=["sepal_length", "petal_length"])
print(X_p.head())


from sklearn.tree import DecisionTreeClassifier

# max_dept = [1, 3, 5, 7]
fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')

ax[0].scatter(data_of_setosa["sepal_length"], data_of_setosa["petal_length"])
ax[0].scatter(data_of_versicolor["sepal_length"],data_of_versicolor["petal_length"])
ax[0].scatter(data_of_virginica["sepal_length"], data_of_virginica["petal_length"])

ax[1].scatter(data_of_setosa["sepal_length"], data_of_setosa["petal_length"])
ax[1].scatter(data_of_versicolor["sepal_length"],data_of_versicolor["petal_length"])
ax[1].scatter(data_of_virginica["sepal_length"], data_of_virginica["petal_length"])

ax[2].scatter(data_of_setosa["sepal_length"], data_of_setosa["petal_length"])
ax[2].scatter(data_of_versicolor["sepal_length"],data_of_versicolor["petal_length"])
ax[2].scatter(data_of_virginica["sepal_length"], data_of_virginica["petal_length"])



X = data[['sepal_length', 'petal_length']]
y = data['species']


model1 = DecisionTreeClassifier(max_depth=6)
model1.fit(X, y)
y1_p = model1.predict(X_p)
ax[0].contourf(x1_p, x2_p, y1_p.reshape(x1_p.shape), alpha=0.3, levels=[0, 1.5, 2.5])

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

model2 = DecisionTreeClassifier(max_depth=6)
bagging = BaggingClassifier(model2, n_estimators=10, max_samples=0.6, random_state=1)
bagging.fit(X, y)
y2_p = bagging.predict(X_p)
ax[1].contourf(x1_p, x2_p, y2_p.reshape(x1_p.shape), alpha=0.3, levels=[0, 1.5, 2.5])

model3 = RandomForestClassifier(max_depth=6, n_estimators=10, max_samples=0.6, random_state=1)
model3.fit(X, y)
y3_p = model3.predict(X_p)
ax[2].contourf(x1_p, x2_p, y3_p.reshape(x1_p.shape), alpha=0.3, levels=[0, 1.5, 2.5])



# for md in max_dept[i]:
#     model = DecisionTreeClassifier(max_depth=md)
#     model.fit(X, y)
#     y_p = model.predict(X_p)
#
#     ax[0,j].scatter(data_of_virginica_A["sepal_length"], data_of_virginica_A["petal_length"])
#     ax[0,j].scatter(data_of_versicolor_A["sepal_length"], data_of_versicolor_A["petal_length"])
#     ax[0,j].contourf(x1_p, x2_p, y_p.reshape(x1_p.shape), alpha=0.3, levels=[0, 1.5, 2.5])
#     j += 1
#
# for md in max_dept[i]:
#     model = DecisionTreeClassifier(max_depth=md)
#     model.fit(X, y)
#     y_p = model.predict(X_p)
#
#     ax[1,j].scatter(data_of_virginica_B["sepal_length"], data_of_virginica_B["petal_length"])
#     ax[1,j].scatter(data_of_versicolor_B["sepal_length"], data_of_versicolor_B["petal_length"])
#     ax[1,j].contourf(x1_p, x2_p, y_p.reshape(x1_p.shape), alpha=0.3, levels=[0, 1.5, 2.5])
#     j += 1

data = iris[['sepal_length', 'petal_length', 'species']]
data_setosa = data[data['species'] == "setosa"]

X = data_setosa['sepal_length']
Y = data_setosa['pepal_length']
data_setosa = data_setosa.drop(columns=['species'])


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data_setosa)
print('1')
print(pca.components_)
print(pca.mean_)
print(pca.explained_variance_)

plt.scatter(X, Y)
plt.scatter(pca.mean_[0], pca.mean_[1])
plt.plot([pca.mean_[0], pca.mean_[0] + pca.components_[0][0] * np.sqrt(pca.explained_variance_[0])],
         [pca.mean_[1], pca.mean_[1] + pca.components_[1][1] * np.sqrt(pca.explained_variance_[1])])


plt.plot([pca.mean_[0], pca.mean_[0] + pca.components_[1][0] * np.sqrt(pca.explained_variance_[0])],
         [pca.mean_[1], pca.mean_[1] + pca.components_[1][1] * np.sqrt(pca.explained_variance_[1])])

pca1 = PCA(n_components=1)
pca1.fit(data_setosa)

X_pca1 = pca1.transform(data_setosa)

print(data_setosa.shape)
print(X_pca1.shape)

X_new = pca1.inverse_transform(X_pca1)
plt.scatter(X_new[:, 0], X_new[:, 1])

plt.plot()


