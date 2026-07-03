import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from SIR_model import SimulationResult


class Visualizer:
    """
    Класс для визуализации результатов моделирования и машинного обучения
    """
    @staticmethod
    def _save_and_show(save_path: str | None) -> None:
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_comparison(
        self,
        trap_result: SimulationResult,
        reference_result: SimulationResult,
        save_path: str | None = None,
    ) -> None:
        """
        Строит сравнение метода трапеций и эталонного решения DOP853.
        """
        plt.figure(figsize=(10, 6))

        plt.plot(trap_result.time, trap_result.susceptible, label="S(t), метод трапеций")
        plt.plot(trap_result.time, trap_result.infectious, label="I(t), метод трапеций")
        plt.plot(trap_result.time, trap_result.removed, label="R(t), метод трапеций")

        plt.plot(reference_result.time, reference_result.susceptible, "--", label="S(t), DOP853")
        plt.plot(reference_result.time, reference_result.infectious, "--", label="I(t), DOP853")
        plt.plot(reference_result.time, reference_result.removed, "--", label="R(t), DOP853")

        plt.xlabel("Время t, с")
        plt.ylabel("Доли распределения")
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        plt.grid(True)
        plt.legend()
        self._save_and_show(save_path)

    def plot_errors(
        self,
        trap_result: SimulationResult,
        reference_result: SimulationResult,
        save_path: str | None = None,
    ) -> None:
        """
        Строит графики абсолютных ошибок по компонентам S, I, R
        """
        error_s = np.abs(trap_result.susceptible - reference_result.susceptible)
        error_i = np.abs(trap_result.infectious - reference_result.infectious)
        error_r = np.abs(trap_result.removed - reference_result.removed)

        plt.figure(figsize=(10, 6))
        plt.plot(trap_result.time, error_s, label="S(t)")
        plt.plot(trap_result.time, error_i, label="I(t)")
        plt.plot(trap_result.time, error_r, label="R(t)")

        plt.xlabel("Время t, с")
        plt.ylabel("Абсолютная ошибка")
        plt.grid(True)
        plt.legend()
        self._save_and_show(save_path)

    def plot_population_sum(
        self,
        result: SimulationResult,
        save_path: str | None = None,
    ) -> None:
        """
        Строит график суммы S(t) + I(t) + R(t)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(result.time, result.population_sum(), label="S(t) + I(t) + R(t)")
        plt.axhline(1.0, linestyle="--", label="Идеальное значение 1")

        plt.xlabel("Время t")
        plt.ylabel("Полная популяция")
        plt.grid(True)
        plt.legend()
        plt.ylim(1 - 1e-14, 1 + 1e-14)
        self._save_and_show(save_path)

    def plot_ml_predictions(
        self,
        y_test: pd.DataFrame,
        y_pred: np.ndarray,
        target_name: str,
        save_path: str | None = None,
    ) -> None:
        """
        Строит график фактических и предсказанных значений для одной целевой переменной
        """
        if target_name not in y_test.columns:
            raise ValueError(f"В y_test нет столбца {target_name}.")

        target_index = list(y_test.columns).index(target_name)

        true_values = y_test[target_name].to_numpy()
        predicted_values = y_pred[:, target_index]

        plt.figure(figsize=(7, 7))
        plt.scatter(true_values, predicted_values)

        min_value = min(float(true_values.min()), float(predicted_values.min()))
        max_value = max(float(true_values.max()), float(predicted_values.max()))
        plt.plot([min_value, max_value], [min_value, max_value], "--", label="Идеальный прогноз")

        plt.xlabel("Фактическое значение")
        plt.ylabel("Предсказанное значение")
        plt.title(f"Качество прогноза: {target_name}")
        plt.grid(True)
        plt.legend()
        self._save_and_show(save_path)
