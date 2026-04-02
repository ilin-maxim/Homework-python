
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# Субграфики
# ax1 = plt.axes()
# ax1.plot(np.sin(x))
# # Нижний, левый, ширина, высота
# #
# ax2 = plt.axes([0.4, 0.3, 0.2, 0.1])
# ax2.plot(np.cos(x))


# fig = plt.figure()
# ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
# ax2 = fig.add_axes([0.4, 0.3, 0.2, 0.1])
# ax1.plot(np.sin(x))
# ax2.plot(np.cos(x))

# for i in range(1, 7):
#     ax = fig.add_subplot(2, 3, i)
#     ax.plot(np.sin(x + np.pi/4 * i))
# x1 = np.linspace(0, 10, 50)
# x2 = np.linspace(0, 20, 50)
#
#
# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
# for i in range(2):
#     for j in range(3):
#         if i % 2 == 0:
#             ax[i, j].plot(np.sin(x1 + np.pi / 4 * (i * 2 + j)))
#         else:
#             ax[i, j].plot(np.sin(x2 + np.pi / 4 * (i * 2 + j)))

# grid = plt.GridSpec(2, 3, wspace=0.2, hspace=0.2)

# plt.subplot(grid[0, 0])
# plt.subplot(grid[0, 1])
# plt.subplot(grid[1, :2])
# plt.subplot(grid[:, 2])

# grid = plt.GridSpec(4, 4, wspace=0.2, hspace=0.2)
#
# rng = np.random.default_rng(1)
# x, y = rng.multivariate_normal([0, 0], [[1, 2], [3, 4]], 1000).T
#
# main_axes = plt.subplot(grid[:3, 1:], yticklabels=[])
# y_axes = plt.subplot(grid[:3, 0])
# z_axes = plt.subplot(grid[-1, 1:])
#
# main_axes.plot(x, y, 'ok', alpha=0.2)
# y_axes.hist(y, 40, orientation='horizontal', color="grey")
# y_axes.invert_xaxis()
#
# z_axes.hist(y, 40, color="grey")
# z_axes.invert_yaxis()

# births = pd.read_csv('births.csv')
# births['day'] = births['day'].astype(int)
# births.index = pd.to_datetime(
#     births['year'] * 1000 + births['month'] * 100 + births['day'], format('Y%m%d')
# )
# births_dom = births.pivot_table('births', index=[births.index.month, births.index.day])
# fig, ax = plt.subplots()
# births_dom.index = [datetime(1969, month, day) for (month, day) in births_dom.index]
# births_dom.plot(ax=ax)
#
# ax.text('1969-10-10', 5500, 'Новый год', )
# ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=15))
# ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%h'))
#
# ax.annotate('Текст аннотации', xy=('1969-1-1', 5500), xytext=('1969-12-1, 4500'),
#             arrowprops=dict(arrowstyle='->', facecolor='black', connectionstyle='angle3,angleA=0,angleB=90'))

fig = plt.figure()
ax1 = plt.axes()
ax1.set_xlim(0, 2)
ax2 = plt.axes([0.4, 0.3, 0.1, 0.2])

ax1.text(0.6, 0.8, '#1_1', transform=ax1.transData)
ax2.text(0.6, 0.8, '#2_1', transform=ax2.transData)

ax1.text(0.5, 0.1, '#1_2', transform=ax1.transAxes)
ax2.text(0.5, 0.1, '#2_2', transform=ax2.transAxes)

ax1.text(0.1, 0.1, '#1_3', transform=fig.transFigure)
ax2.text(0.2, 0.2, '#2_3', transform=fig.transFigure)

plt.show()








