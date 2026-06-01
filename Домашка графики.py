import numpy as np
import matplotlib.pyplot as plt


fig1, ax1 = plt.subplots()
x = np.array([1, 5, 10, 15, 20])
y_1 = np.array([1, 7, 3, 5, 11])
y_2 = np.array([4, 3, 1, 8, 12])

ax1.plot(x, y_1, '-o', color='red', linewidth=2, markersize=8, label='Зависимость 1')
ax1.plot(x, y_2, '-.o', color='green', linewidth=2, markersize=8, label='Зависимость 2')

ax1.set_title('График 1')
ax1.set_xlabel('Ось x')
ax1.set_ylabel('Ось y')
ax1.legend()


fig2 = plt.figure(figsize=(10, 5))
grid = plt.GridSpec(2, 2, hspace=0.35, wspace=0.25)
ax2_1 = plt.subplot(grid[0, :])
ax2_2 = plt.subplot(grid[1, 0])
ax2_3 = plt.subplot(grid[1, 1])

x = np.array([1, 2, 3, 4, 5])
y_1 = np.array([1, 7, 6, 3, 5])
y_2 = np.array([9, 4, 2, 4, 9])
y_3 = np.array([-7, -4, 2, -4, -7])

ax2_1.plot(x, y_1)
ax2_2.plot(x, y_2)
ax2_3.plot(x, y_3)

ax2_1.set_title('График 1')
ax2_2.set_title('График 2')
ax2_3.set_title('График 3')


fig3, ax3 = plt.subplots()

x = np.linspace(-5, 5, 50)
y = x ** 2

ax3.plot(x, y, linewidth=2)
ax3.annotate(
    'min',
    xy=(0, 0),
    xytext=(0, 10),
    ha='center',
    arrowprops=dict(facecolor='green', edgecolor='black', shrink=0.05, width=6)
)

ax3.set_title('Минимум функции')
ax3.set_xlabel('x')
ax3.set_ylabel('y')


fig4, ax4 = plt.subplots()

rng = np.random.default_rng(0)
data = rng.integers(0, 11, size=(7, 7))

image = ax4.imshow(data, cmap='viridis', origin='lower', extent=[0, 7, 0, 7])
fig4.colorbar(image, ax=ax4)

ax4.set_title('Цветовая схема')
ax4.set_xlabel('Горизонтальная ось')
ax4.set_ylabel('Вертикальная ось')


fig5, ax5 = plt.subplots()

x = np.linspace(0, 5, 300)
y = np.cos(np.pi * x)

ax5.fill_between(x, 0, y)
ax5.plot(x, y, color='red', linewidth=2)

ax5.set_title('График косинуса')
ax5.set_xlabel('Ось x')
ax5.set_ylabel('Ось y')


fig6, ax6 = plt.subplots()

x = np.linspace(0, 5, 300)
y = np.cos(np.pi * x)
y_cut = y.copy()
y_cut[y_cut < -0.5] = np.nan

ax6.plot(x, y_cut, linewidth=2)

ax6.set_title('График косинуса с пропусками')
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_ylim(-1, 1)


fig7, ax7 = plt.subplots(1, 3)

x = np.arange(0, 7)
y = np.arange(0, 7)

ax7[0].step(x, y, where='pre', color='green', linewidth=2)
ax7[0].plot(x, y, 'go')

ax7[1].step(x, y, where='post', color='green', linewidth=2)
ax7[1].plot(x, y, 'go')

ax7[2].step(x, y, where='mid', color='green', linewidth=2)
ax7[2].plot(x, y, 'go')

ax7[0].grid(True)
ax7[1].grid(True)
ax7[2].grid(True)


fig8, ax8 = plt.subplots()

x = np.arange(0, 11)
y1 = np.array([0, 1.5, 3, 4.2, 4.8, 5, 4.7, 4.2, 3.2, 1.7, 0])
y2 = np.array([0, 3, 6.5, 8.5, 9.7, 10, 9.6, 8.4, 6, 3, 0])
y3 = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

ax8.stackplot(x, y1, y2, y3, labels=['y_1', 'y_2', 'y_3'])
ax8.legend(loc='upper left')

ax8.set_title('График площадей')
ax8.set_xlabel('Ось x')
ax8.set_ylabel('Ось y')


fig9, ax9 = plt.subplots()

labels = ['Ford', 'Toyota', 'BMW', 'Audi', 'Jaguar']
sizes = [15, 10, 35, 17, 23]
explode = [0, 0, 0.15, 0, 0]

ax9.pie(sizes, labels=labels, explode=explode, startangle=0)


fig10, ax10 = plt.subplots()

labels = ['Ford', 'Toyota', 'BMW', 'Audi', 'Jaguar']
sizes = [15, 10, 35, 17, 23]

ax10.pie(sizes, labels=labels, startangle=0, wedgeprops=dict(width=0.4))


plt.show()