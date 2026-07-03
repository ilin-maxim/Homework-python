from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


ArrayFloat = npt.NDArray[np.float64]


@dataclass
class SIRModel:
    """
    Класс, описывающий математическую SIR-модель
    beta  — коэффициент заражения;
    gamma — коэффициент выздоровления;
    s0    — начальная доля восприимчивых;
    i0    — начальная доля инфицированных;
    r0    — начальная доля выздоровевших.
    """
    beta: float
    gamma: float
    s0: float
    i0: float
    r0: float

    def validate(self) -> None:
        """
        Метод проверки параметров модели
        """
        if self.beta <= 0:
            raise ValueError("Коэффициент заражения beta должен быть положительным")

        if self.gamma <= 0:
            raise ValueError("Коэффициент выздоровления gamma должен быть положительным")

        initial_values = [self.s0, self.i0, self.r0]
        if any(value < 0 for value in initial_values):
            raise ValueError("Начальные доли S_0, I_0, R_0 не могут быть отрицательными.")

        initial_sum = self.s0 + self.i0 + self.r0
        if not np.isclose(initial_sum, 1.0, atol=1e-10):
            raise ValueError(
                "Сумма начальных долей S_0 + I_0 + R_0 должна быть равна 1. "
                f"Сейчас получено: {initial_sum}"
            )

    @property
    def initial_state(self) -> ArrayFloat:
        """
        Возвращает начальное состояние системы в виде numpy-массива
        """
        return np.array([self.s0, self.i0, self.r0], dtype=np.float64)

    def right_part(self, state: ArrayFloat) -> ArrayFloat:
        """
        Вычисляет правую часть системы SIR:
        dS/dt = - beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I
        """
        susceptible = float(state[0])
        infectious = float(state[1])

        ds_dt = -self.beta * susceptible * infectious
        di_dt = self.beta * susceptible * infectious - self.gamma * infectious
        dr_dt = self.gamma * infectious

        return np.array([ds_dt, di_dt, dr_dt], dtype=np.float64)

    def reproduction_number(self) -> float:
        """
        Возвращает базовый показатель распространения R0 = beta / gamma.
        Здесь это именно эпидемиологический показатель, а не начальное значение R(0).
        """
        return self.beta / self.gamma


@dataclass
class SimulationResult:
    """
    Класс для хранения результата численного моделирования
    """
    time: ArrayFloat
    susceptible: ArrayFloat
    infectious: ArrayFloat
    removed: ArrayFloat

    def to_dataframe(self) -> pd.DataFrame:
        """
        Переводит результат моделирования в pandas.DataFrame
        """
        return pd.DataFrame(
            {
                "time": self.time,
                "susceptible": self.susceptible,
                "infectious": self.infectious,
                "removed": self.removed,
                "population_sum": self.population_sum(),
            }
        )

    def population_sum(self) -> ArrayFloat:
        """
        Возвращает сумму S(t) + I(t) + R(t) на всей временной сетке
        """
        return self.susceptible + self.infectious + self.removed

    def max_infected(self) -> float:
        """
        Возвращает максимальную долю инфицированных
        """
        return float(np.max(self.infectious))

    def time_to_peak(self) -> float:
        """
        Возвращает время, в которое достигается максимум инфицированных
        """
        return float(self.time[int(np.argmax(self.infectious))])

    def final_removed(self) -> float:
        """
        Возвращает итоговую долю выздоровевших к концу моделирования
        """
        return float(self.removed[-1])

    def max_population_deviation(self) -> float:
        """
        Возвращает максимальное отклонение суммы S + I + R от 1
        """
        return float(np.max(np.abs(self.population_sum() - 1.0)))
