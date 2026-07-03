import numpy as np
import pandas as pd

from Decorators import log_call, measure_time
from SIR_model import SIRModel
from Solvers import TrapezoidNewtonSolver


class ExperimentRunner:
    """
    Класс для запуска серии вычислительных экспериментов
    """
    def __init__(
        self,
        t_start: float,
        t_finish: float,
        step: float,
        epsilon: float,
        max_iterations: int,
    ) -> None:
        self.t_start = t_start
        self.t_finish = t_finish
        self.step = step
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def run_single_scenario(
        self,
        beta: float,
        gamma: float,
        i0: float,
        r0: float = 0.0,
    ) -> dict[str, float]:
        """
        Запускает один сценарий и возвращает основные численные характеристики
        """
        s0 = 1.0 - i0 - r0

        model = SIRModel(
            beta=np.float64(beta),
            gamma=float(gamma),
            s0=float(s0),
            i0=float(i0),
            r0=float(r0),
        )

        solver = TrapezoidNewtonSolver(
            model=model,
            t_start=self.t_start,
            t_finish=self.t_finish,
            step=self.step,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations,
        )

        result = solver.solve()

        return {
            "beta": float(beta),
            "gamma": float(gamma),
            "reproduction_number": float(model.reproduction_number()),
            "s0": float(s0),
            "i0": float(i0),
            "r0": float(r0),
            "max_infected": result.max_infected(),
            "time_to_peak": result.time_to_peak(),
            "final_removed": result.final_removed(),
            "max_population_deviation": result.max_population_deviation(),
        }

    @measure_time
    @log_call
    def run_grid(
        self,
        beta_values: np.ndarray | None = None,
        gamma_values: np.ndarray | None = None,
        i0_values: np.ndarray | None = None,
        csv_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Запускает сетку экспериментов
        """
        if beta_values is None:
            beta_values = np.linspace(0.15, 0.80, 8)

        if gamma_values is None:
            gamma_values = np.linspace(0.05, 0.30, 6)

        if i0_values is None:
            i0_values = np.linspace(0.01, 0.10, 6)

        rows: list[dict[str, float]] = []

        for beta in beta_values:
            for gamma in gamma_values:
                for i0 in i0_values:
                    rows.append(
                        self.run_single_scenario(
                            beta=float(beta),
                            gamma=float(gamma),
                            i0=float(i0),
                        )
                    )

        dataframe = pd.DataFrame(rows)

        if csv_path is not None:
            dataframe.to_csv(csv_path, index=False, encoding="utf-8")

        return dataframe
