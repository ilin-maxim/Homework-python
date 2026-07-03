from Config import (
    BETA,
    GAMMA,
    I_0,
    R_0,
    S_0,
    EXPERIMENTS_CSV_PATH,
    NEWTON_EPSILON,
    NEWTON_MAX_ITERATIONS,
    POLYNOMIAL_DEGREE,
    STEP,
    T_FINISH,
    T_START,
)
from Experiments import ExperimentRunner
from ML_analysis import MLAnalyzer
from SIR_model import SIRModel
from Solvers import ReferenceSolver, TrapezoidNewtonSolver
from Visualization import Visualizer


def print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """
    Печатает метрики качества ML-модели
    """
    print("\nМетрики полиномиальной регрессии:")

    for target_name, target_metrics in metrics.items():
        print(f"\nЦелевая переменная: {target_name}")

        for metric_name, metric_value in target_metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")


def main() -> None:
    """
    Главный сценарий работы программы
    """
    model = SIRModel(
        beta=BETA,
        gamma=GAMMA,
        s0=S_0,
        i0=I_0,
        r0=R_0,
    )

    trapezoid_solver = TrapezoidNewtonSolver(
        model=model,
        t_start=T_START,
        t_finish=T_FINISH,
        step=STEP,
        epsilon=NEWTON_EPSILON,
        max_iterations=NEWTON_MAX_ITERATIONS,
    )

    reference_solver = ReferenceSolver(
        model=model,
        t_start=T_START,
        t_finish=T_FINISH,
        step=STEP,
    )

    trap_result = trapezoid_solver.solve()
    reference_result = reference_solver.solve()

    visualizer = Visualizer()
    visualizer.plot_comparison(trap_result, reference_result)
    visualizer.plot_errors(trap_result, reference_result)
    visualizer.plot_population_sum(trap_result)

    experiment_runner = ExperimentRunner(
        t_start=T_START,
        t_finish=T_FINISH,
        step=STEP,
        epsilon=NEWTON_EPSILON,
        max_iterations=NEWTON_MAX_ITERATIONS,
    )

    experiments_df = experiment_runner.run_grid(csv_path=EXPERIMENTS_CSV_PATH)

    ml_analyzer = MLAnalyzer(degree=POLYNOMIAL_DEGREE)
    metrics = ml_analyzer.train(experiments_df)
    print_metrics(metrics)

    new_i0 = 0.05
    new_r0 = 0.0
    new_s0 = 1.0 - new_i0 - new_r0

    prediction = ml_analyzer.predict(
        beta=0.45,
        gamma=0.12,
        s0=new_s0,
        i0=new_i0,
        r0=new_r0,
    )

    print("\nПример ML-прогноза для нового сценария:")
    print(f"beta = 0.45, gamma = 0.12, S0 = {new_s0:.2f}, I0 = {new_i0:.2f}, R0 = {new_r0:.2f}")
    print(
        "Предсказанная максимальная доля инфицированных: "
        f"{prediction['predicted_max_infected']:.6f}"
    )
    print(
        "Предсказанное время наступления пика: "
        f"{prediction['predicted_time_to_peak']:.6f}"
    )

    if ml_analyzer.y_test is not None and ml_analyzer.y_pred is not None:
        visualizer.plot_ml_predictions(
            y_test=ml_analyzer.y_test,
            y_pred=ml_analyzer.y_pred,
            target_name="max_infected",
        )
        visualizer.plot_ml_predictions(
            y_test=ml_analyzer.y_test,
            y_pred=ml_analyzer.y_pred,
            target_name="time_to_peak",
        )


if __name__ == "__main__":
    main()
