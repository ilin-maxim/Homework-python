import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.ticker as mtick
from mpl_toolkits import mplot3d

grid = plt.GridSpec(1, 2)
ax1 = plt.subplot(grid[0, 0])
ax1.set_xscale('log')
ax1.set_xlim(1, 1000)
ax1.grid(True, which='major')

ax2 = plt.subplot(grid[0, 1])
ax2.set_yscale('log')
ax2.set(ylim=(1,1000))
ax2.grid(True, which='minor', axis='y')

print(ax1.xaxis.get_major_locator())
print(ax1.xaxis.get_major_formatter())
print(ax1.xaxis.get_minor_locator())
print(ax1.xaxis.get_minor_formatter())

print(ax1.yaxis.get_major_locator())
print(ax1.yaxis.get_major_formatter())
print(ax1.yaxis.get_minor_locator())
print(ax1.yaxis.get_minor_formatter())

ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax2.xaxis.set_major_locator(plt.NullLocator())

faces = fetch_olivetti_faces().images
fig, ax = plt.subplot(7, 7)
fig.subplots_adjust(hspace=0, wspace=0)
for i in range(7):
    for j in range(7):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[7*i + j])

def ff(value, tick_number):
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return 0
    elif N == 1:
        return "$\frac{\pi}{2}$"
    elif N == 2:
        return "$\pi$"
    elif N%2 == 0:
        t = N / 2
        return f"{t}" + r"$\pi$"
    else:
        return f"{N}" + r"$\frac{\pi}{2}$"

x = np.linspace(0, 4*np.pi, 1000)

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)

ax.plot(x, np.sin(x), label='sin')
ax.plot(x, np.cos(x), label='cos')

# for a in ax.flat:
#     a.xaxis.set_major_locator(plt.MaxNLocator(10))
#     a.yaxis.set_major_locator(plt.MaxNLocator(2))

ax.legend()
ax.grid(True)
ax.set_xlim(0, 4*np.pi)
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/4))

ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))

fig, ax = plt.subplots(8, 1)
x = np.linspace(0, 10, 20)
for i in ax.flat:
    i.plot(x, x * 0 + 2)

ax[0].xaxis.set_major_locator(plt.NullLocator())
ax[1].xaxis.set_major_locator(plt.MultipleLocator(0.8))
ax[2].xaxis.set_major_locator(plt.FixedLocator([1, 3, 8, 9]))
ax[3].xaxis.set_major_locator(plt.LinearLocator(4))
ax[4].xaxis.set_major_locator(plt.IndexLocator(base=1.5, offset=0.3))
ax[5].xaxis.set_major_locator(plt.AutoLocator())
ax[6].xaxis.set_major_locator(plt.MaxNLocator(8))
ax[7].xaxis.set_major_locator(plt.LogLocator(base=3))

ax[1].xaxis.set_major_formatter(plt.NullFormatter())
ax[2].xaxis.set_major_formatter(plt.FixedFormatter(['a', 'b', 'c', 'd']))
ax[3].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f $m^2$'))
ax[4].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=4))


x = np.random.rand(1000)
plt.style.use('lec_13.style')
# plt.axes(facecolor='#adadad')
# plt.figure(facecolor='#921212')
plt.rc("figure", facecolor='#921212')
plt.rc("axes", facecolor='#adadad')
print(plt.style.available)
plt.style.use('grayscale')
plt.hist(x)

def f(x,y):
    return np.sin(x * np.pi / 2 + np.sqrt(x**2 + y**2))
fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(-6, 6, 30)
y = np.linspace(-10, 10, 50)

print(x.shape)
print(y.shape)

X, Y = np.meshgrid(x, y)
print(X.shape)
print(Y.shape)
print(X)
print(Y)

Z = f(X, Y)
print(Z.shape)
print(Z)

ax.contour3D(X, Y, Z, 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(30, 90)




plt.show()