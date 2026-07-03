from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from Decorators import log_call, measure_time, validate_solver_model
from SIR_model import SIRModel, SimulationResult


ArrayFloat = npt.NDArray[np.float64]


class BaseSolver(ABC):
    """
    Абстрактный базовый класс для всех решателей
    """

    @abstractmethod
    def solve(self) -> SimulationResult:
        pass


class TrapezoidNewtonSolver(BaseSolver):
    """
    Решатель SIR-модели методом трапеций.
    Так как метод трапеций является неявным, на каждом временном шаге
    нелинейная система решается методом Ньютона.
    """

    def __init__(
        self,
        model: SIRModel,
        t_start: float,
        t_finish: float,
        step: float,
        epsilon: float = 1e-7,
        max_iterations: int = 100,
    ) -> None:
        self.model = model
        self.t_start = t_start
        self.t_finish = t_finish
        self.step = step
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def residual(self, next_state: ArrayFloat, current_state: ArrayFloat) -> ArrayFloat:
        """
        Вычисляет вектор невязки метода трапеций:
        F(y_{n+1}) = y_{n+1} - y_n - h/2 * (f(y_n) + f(y_{n+1}))
        """
        current_sir_values = self.model.right_part(current_state)
        next_sir_values = self.model.right_part(next_state)

        return (
            next_state
            - current_state
            - 0.5 * self.step * (current_sir_values + next_sir_values)
        )

    def jacobian(self, next_state: ArrayFloat) -> ArrayFloat:
        """
        Вычисляет матрицу Якоби для вектора невязки
        """
        susceptible = float(next_state[0])
        infectious = float(next_state[1])
        beta = self.model.beta
        gamma = self.model.gamma
        h = self.step

        return np.array(
            [
                [
                    1.0 + 0.5 * h * beta * infectious,
                    0.5 * h * beta * susceptible,
                    0.0,
                ],
                [
                    -0.5 * h * beta * infectious,
                    1.0 - 0.5 * h * (beta * susceptible - gamma),
                    0.0,
                ],
                [
                    0.0,
                    -0.5 * h * gamma,
                    1.0,
                ],
            ],
            dtype=np.float64
        )

    def newton_step(self, current_state: ArrayFloat, step_index: int) -> ArrayFloat:
        """
        Метод Ньютона для нахождения состояния системы на следующем шаге.
        Начальное приближение берётся равным текущему состоянию.
        """
        current_guess = current_state.copy()

        for iteration in range(self.max_iterations):
            current_residual = self.residual(current_guess, current_state)

            if np.linalg.norm(current_residual, ord=np.inf) < self.epsilon:
                return current_guess

            try:
                delta_guess = np.linalg.solve(
                    self.jacobian(current_guess),
                    -current_residual,
                )
            except np.linalg.LinAlgError as error:
                raise RuntimeError(
                    f"На шаге n = {step_index} метод Ньютона остановлен: "
                    f"матрица Якоби вырождена на итерации {iteration}."
                ) from error

            current_guess = current_guess + delta_guess

            if np.linalg.norm(delta_guess, ord=np.inf) < self.epsilon:
                return current_guess

        raise RuntimeError(
            f"Метод Ньютона не сошёлся за {self.max_iterations} итераций "
            f"на шаге n = {step_index}."
        )

    @measure_time
    @log_call
    @validate_solver_model
    def solve(self) -> SimulationResult:
        """
        Основной цикл метода трапеций по временной сетке
        """
        node_count = int(round((self.t_finish - self.t_start) / self.step)) + 1
        time_grid = np.linspace(
            self.t_start,
            self.t_finish,
            node_count,
            dtype=np.float64,
        )

        solution = np.zeros((node_count, 3), dtype=np.float64)
        solution[0] = self.model.initial_state

        for step_index in range(node_count - 1):
            solution[step_index + 1] = self.newton_step(
                current_state=solution[step_index],
                step_index=step_index,
            )

        return SimulationResult(
            time=time_grid,
            susceptible=solution[:, 0],
            infectious=solution[:, 1],
            removed=solution[:, 2],
        )


class ReferenceSolver(BaseSolver):
    """
    Эталонный решатель через scipy.integrate.solve_ivp с методом DOP853.
    Используется для проверки точности метода трапеций.
    """
    def __init__(
        self,
        model: SIRModel,
        t_start: float,
        t_finish: float,
        step: float,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        self.model = model
        self.t_start = t_start
        self.t_finish = t_finish
        self.step = step
        self.rtol = rtol
        self.atol = atol

    @measure_time
    @log_call
    @validate_solver_model
    def solve(self) -> SimulationResult:
        """
        Получает эталонное решение методом DOP853
        """
        node_count = int(round((self.t_finish - self.t_start) / self.step)) + 1
        time_grid = np.linspace(
            self.t_start,
            self.t_finish,
            node_count,
            dtype=np.float64,
        )

        solution = solve_ivp(
            fun=lambda time, state: self.model.right_part(
                np.asarray(state, dtype=np.float64)
            ),
            t_span=(float(time_grid[0]), float(time_grid[-1])),
            y0=self.model.initial_state,
            method="DOP853",
            t_eval=time_grid,
            rtol=self.rtol,
            atol=self.atol,
        )

        if not solution.success:
            raise RuntimeError("Не удалось получить эталонное решение через solve_ivp.")

        solution_y = np.asarray(solution.y, dtype=np.float64)

        return SimulationResult(
            time=time_grid,
            susceptible=solution_y[0],
            infectious=solution_y[1],
            removed=solution_y[2],
        )
