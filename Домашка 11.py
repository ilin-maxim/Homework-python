import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_digist
from sklearn.manifold import Isomap

x = np.linspace(0, 10, 1000)


# fig, ax = plt.subplots()
#
# ax.plot(x, np.sin(x), '-k', label='sin(x)')
# ax.plot(x, np.cos(x), '--r', label='cos(x)')
# ax.axis('equal')
#
# ax.legend(frameon=False, shadow=True, borderpad=1, loc='lower center', ncol=2)
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
# lines = plt.plot(x, y)
# plt.legend(lines[:2], ['Первая', 'Вторая', 'Третья'])
# plt.plot(x, y[:,0], label='Первый')
# plt.plot(x, y[:,1], label='Второй')
# plt.plot(x, y[:,2])
# plt.legend()

# cities = pd.read_csv('california_cities.csv')
# latd = cities['latd']
# longd = cities['longd']
# population_total = cities['population_total']
# area_total_km2 = cities['area_total_km2']
#
# plt.scatter(latd, longd, c=np.log10(population_total), s=area_total_km2, alpha=0.5)
#
# plt.scatter([],[], s=100, c='k', labek='100 $km^2$', alpha=0.5)
# plt.scatter([],[], s=300, c='k', labek='100 $km^2$', alpha=0.5)
# plt.scatter([],[], s=500, c='k', labek='100 $km^2$', alpha=0.5)
# plt.legend(frameon=False, labelspacing=2, title='Площадь')
#
# plt.colorbar()
# plt.axis('equal')

# fig, ax = plt.subplots()
# lines = ax.plot(x, np.sin(x[:, np.newaxis] - np.pi / 2 * np.arange(0, 4, 0.5)))
# ax.axis('equal')
# ax.legend(lines[:2], ['Line A', 'Line B'], loc='lower right')
# leg = mpl.legend.Legend(ax, lines[2:], ['Line C', 'Line D'], loc='upper right')
# leg2 = mpl.legend.Legend(ax, lines[2:], ['Line C', 'Line D'], loc='upper left')
# ax.add_artist(leg)
# ax.add_artist(leg2)

y = np.sin(x) * np.cos(x[:, np.newaxis])


# plt.imshow(y, cmap='jet')
# plt.colorbar()

digits = load_digist(n_class=6)
print(digits)
# fig, ax = plt.subplots(8, 8)
#
# for i, ax_ enumerate(ax.flax):
#     ax_.imshow(digits.images[i], cmap='binary')
#     ax_.set(xticks=[], yticks=[])

iso = Isomap(n_components=2, n_neighbours=10)
prj = iso.fit_transform(digits.data)
plt.scatter(
    prj[:, 0],
    prj[:, 1],
    c=digits.target,
    cmap=plt.cm.get_cmap("jet", 6)
)
plt.colorbar(ticks=range(6))
plt.clim(-0.5, 5.5)

plt.show()



