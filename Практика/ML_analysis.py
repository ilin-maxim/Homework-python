import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from Decorators import log_call, measure_time


class MLAnalyzer:
    """
    Класс для обучения модели машинного обучения
    """
    def __init__(self, degree: int = 2, test_size: float = 0.25, random_state: int = 42) -> None:
        self.degree = degree
        self.test_size = test_size
        self.random_state = random_state

        self.feature_columns = [
            "beta",
            "gamma",
            "reproduction_number",
            "s0",
            "i0",
            "r0",
        ]

        self.target_columns = [
            "max_infected",
            "time_to_peak",
        ]

        self.model: Pipeline | None = None
        self.x_test: pd.DataFrame | None = None
        self.y_test: pd.DataFrame | None = None
        self.y_pred: np.ndarray | None = None

    @measure_time
    @log_call
    def train(self, dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Обучает полиномиальную регрессию и возвращает метрики качества
        """
        x = dataframe[self.feature_columns]
        y = dataframe[self.target_columns]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.model = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("regression", LinearRegression()),
            ]
        )

        self.model.fit(x_train, y_train)

        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = self.model.predict(x_test)

        return self.evaluate()

    def evaluate(self) -> dict[str, dict[str, float]]:
        """
        Считает MAE, MSE, RMSE и R2 отдельно для каждого целевого признака
        """
        if self.y_test is None or self.y_pred is None:
            raise RuntimeError("Сначала нужно обучить модель методом train()")

        metrics: dict[str, dict[str, float]] = {}

        for target_index, target_name in enumerate(self.target_columns):
            true_values = self.y_test[target_name].to_numpy()
            predicted_values = self.y_pred[:, target_index]

            mse = mean_squared_error(true_values, predicted_values)

            metrics[target_name] = {
                "MAE": float(mean_absolute_error(true_values, predicted_values)),
                "MSE": float(mse),
                "RMSE": float(np.sqrt(mse)),
                "R2": float(r2_score(true_values, predicted_values)),
            }

        return metrics

    def predict(
        self,
        beta: float,
        gamma: float,
        s0: float,
        i0: float,
        r0: float,
    ) -> dict[str, float]:
        """
        Делает прогноз для нового сценария
        """
        if self.model is None:
            raise RuntimeError("Сначала нужно обучить модель методом train()")

        row = pd.DataFrame(
            [
                {
                    "beta": beta,
                    "gamma": gamma,
                    "reproduction_number": beta / gamma,
                    "s0": s0,
                    "i0": i0,
                    "r0": r0,
                }
            ]
        )

        prediction = self.model.predict(row)[0]

        return {
            "predicted_max_infected": float(prediction[0]),
            "predicted_time_to_peak": float(prediction[1]),
        }
