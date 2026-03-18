from pydoc import describe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(pd.date_range("2026-01-01", periods=4, freq="ME"))
print(pd.date_range("2026-01-01", periods=4, freq="MS"))
print(pd.date_range("2026-01-01", periods=4, freq="QE"))
print(pd.date_range("2026-01-01", periods=4, freq="QS"))
print(pd.date_range("2026-01-01", periods=4, freq="W"))
print(pd.date_range("2026-01-01", periods=4, freq="W-MON"))
print(pd.date_range("2026-01-01", periods=4, freq="4W-MON"))

ind = pd.read_csv("index.csv", sep=";", parse_dates=["Date"])
print(ind.head())
print(type(ind))
print(ind.dtypes)

index = pd.DatetimeIndex(ind["Date"])

ind.index = index
ind = ind["Close"]
print(ind.head())

# ind.plot()

df = pd.read_csv("FremontBridge.csv",
                 index_col="Date", parse_dates=True,
                 # date_format="%Y-%m-%d %I:%M:%S %p",
                 date_format="%m/%d/%Y %I:%M:%S %p"
                 )
print(df.head())
print(df.dtypes)

print(df.columns)
df.columns = ["Total", "East", "West"]
print(df.head())
print(df.describe())
print(df.dropna().describe())

# df.plot()
# plt.ylabel("Количество велосипедисов в час")
# plt.show()

# weekly = df.resample("W").sum()
# weekly.plot(style=["-", ":", "--"])
# plt.ylabel("Количество велосипедисов (По неделям)")
# plt.show()

daily = df.resample("W").sum()
# center = False -> прошлые значения от выбранного
# center = True -> прошлые и будущие значения от выбранного
# daily.rolling(30, center=True).mean()
# daily.plot(style=["-", ":", "--"])
# plt.ylabel("Среднемесячное количество велосипедистов (скользящее окно)")
# plt.show()

# daily.rolling(30, center=True, win_type="gaussian").mean(std=5)
# daily.plot(style=["-", ":", "--"])
# plt.ylabel("Среднемесячное количество велосипедистов (скользящее окно)")
# plt.show()

# timely = df.groupby(df.index.time).mean()
# ticks = 60 * 60 * 4 * np.arange(6)
# print(ticks)
# timely.plot(xticks=ticks)
# plt.show()

# weekly = df.groupby(df.index.dayofweek).mean()
# weekly.plot()
# plt.show()

w1 = np.where(df.index.weekday < 5, "Будни", "Выходные")
t1 = df.groupby([w1, df.index.time]).mean()
fig,ax = plt.subplots(1, 2)
ax[0].set_ylim(0, 600)
ax[1].set_ylim(0, 600)
t1.loc["Будни"].plot(ax=ax[0], title="Будни")
t1.loc["Выходные"].plot(ax=ax[1], title="Выходные")
plt.show()

# timely = df.groupby(df.index.time).mean()
# ticks = 60 * 60 * 4 * np.arange(6)
# print(ticks)
# timely.plot(xticks=ticks)
# plt.show()

# MATPLOTLIB

# pip install matplotlib
# pip install pyqt5

plt.style.use('classic')

plt.show() # Должно быть в единственным экземпляре
# fig.savefig('ttt.png')
print(fig.canvas.get_supported_filetypes())