import time
from functools import wraps
from typing import Any, Callable


def measure_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Декоратор для измерения времени выполнения метода
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        finish_time = time.perf_counter()
        print(f"Время выполнения {func.__qualname__} = {finish_time - start_time} секунд")
        return result
    return wrapper


def log_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Декоратор для логирования запуска и завершения метода
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"Служба {func.__qualname__} запущена")
        result = func(*args, **kwargs)
        print(f"Служба {func.__qualname__} завершена")
        return result
    return wrapper


def validate_solver_model(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Декоратор для проверки корректности модели и параметров решателя.
    Предполагается, что у объекта self есть поле model.
    """
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self, "model"):
            raise AttributeError(
                "validate_solver_model можно применять только к объектам с полем model"
            )

        self.model.validate()

        if hasattr(self, "step") and self.step <= 0:
            raise ValueError("Шаг интегрирования должен быть положительным.")

        if hasattr(self, "t_finish") and hasattr(self, "t_start"):
            if self.t_finish <= self.t_start:
                raise ValueError("Правая граница времени должна быть больше левой.")

        return func(self, *args, **kwargs)
    return wrapper
