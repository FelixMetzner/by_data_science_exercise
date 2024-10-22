#!/usr/bin/env python

import pathlib
from dataclasses import dataclass

import numpy as np
import torch

from by_data_science_exercise.loss_definitions import (
    LossFcnCallableHint,
    conrad_negative_log_likelihood,
    grad_of_conrad_negative_log_likelihood,
)
from by_data_science_exercise.model import MeanModel
from by_data_science_exercise.plotting import plot_distributions, plot_loss, plot_loss_function, save_or_show_image

__all__ = [
    "main",
]


# Default parameters defined in the provided code snippet:
default_n_data: int = 1000
default_poisson_rate: float = 2.0
default_min_stock: int = 0
default_max_stock: int = 10

default_torch_seed: int = 0

default_epochs: int = 100
default_learning_rate: float = 0.01


output_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "result_plots"
assert output_path.is_dir(), output_path


@dataclass(frozen=True)
class Experiment:
    name: str
    sample_size: int
    poisson_rate: float
    min_max_stock: tuple[int, int]
    torch_random_seed: int
    stocks_data: torch.Tensor
    demand_data: torch.Tensor
    sales_data: torch.Tensor
    torch_generator: torch.Generator

    @staticmethod
    def create_experiment(
        name: str,
        sample_size: int,
        poisson_rate: float,
        min_max_stock: tuple[int, int] = (0, 10),
        torch_random_seed: int = 0,
    ) -> "Experiment":
        if sample_size < 0:
            raise ValueError(f"Sample size must be greater than zero, but {sample_size=} was provided.")
        if poisson_rate < 0.0:
            raise ValueError(f"Poisson rate must be greater than or equal zero, but {poisson_rate=} was provided.")

        min_stock, max_stock = min_max_stock
        if not min_stock < max_stock and min_stock >= 0:
            raise ValueError(f"Ill-defined min and max stock values were provided: {min_max_stock=}.")

        _gen: torch.Generator = torch.manual_seed(torch_random_seed)

        _stocks_data: torch.Tensor = torch.randint(low=min_stock, high=max_stock, size=(sample_size,), generator=_gen)
        _demand_data: torch.Tensor = torch.poisson(torch.ones(sample_size) * poisson_rate, generator=_gen)
        _sales_data: torch.Tensor = torch.min(_demand_data, _stocks_data)

        return Experiment(
            name=name,
            sample_size=sample_size,
            poisson_rate=poisson_rate,
            min_max_stock=min_max_stock,
            torch_random_seed=torch_random_seed,
            stocks_data=_stocks_data,
            demand_data=_demand_data,
            sales_data=_sales_data,
            torch_generator=_gen,
        )


def run_training(
    loss_function: LossFcnCallableHint,
    experiment: Experiment,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    print_current_values: bool = False,
) -> float:
    """
    Performs training of MeanModel and returns the (absolute value) of the best mean prediction.


    :return: Absolute value of the best mean prediction.
    """

    my_model = MeanModel().to("cpu")

    my_optimizer = torch.optim.SGD(
        params=my_model.parameters(),
        lr=learning_rate,
    )

    _mean_values: list[float] = []
    _loss_values: list[float] = []

    for i in range(n_epochs):
        my_optimizer.zero_grad()

        my_outputs = my_model(experiment.sample_size)
        _mean_values.append(my_outputs[0].item())

        my_current_loss = loss_function(
            mean_predictions=my_outputs,
            sales=experiment.sales_data,
            stocks=experiment.stocks_data,
        )
        my_current_loss.backward()

        my_optimizer.step()

        _loss_values.append(my_current_loss.item())

        if print_current_values:
            print(
                f"Epoch: {i}, "
                f"Loss: {my_current_loss.item()}, "
                f"Current mean: {my_outputs[0].item()}, "
                f"New mean: {my_model(experiment.sample_size)[0].item()}"
            )

    loss_fig, _ = plot_loss(
        loss_values=_loss_values,
        mean_values=_mean_values,
    )
    save_or_show_image(
        fig=loss_fig,
        figure_file_path=output_path / f"loss_curve_for_exp_{experiment.name}.png",
        overwrite_existing=True,
    )

    return np.abs(_mean_values[-1])


def evaluate_current_experiment(
    experiment: Experiment,
) -> None:
    print(
        f"Current experiment '{experiment.name}' with:"
        f"\n\tsample size = {experiment.sample_size}"
        f"\n\ttrue poisson rate = {experiment.poisson_rate}"
        f"\n\tstocks min and max = {experiment.min_max_stock}"
    )

    dist_fig, _ = plot_distributions(
        demand=experiment.demand_data,
        sales=experiment.sales_data,
        stocks=experiment.stocks_data,
        show_uncert=True,
    )
    save_or_show_image(
        fig=dist_fig,
        figure_file_path=output_path / f"data_distributions_for_exp_{experiment.name}.png",
        overwrite_existing=True,
    )


def evaluate_loss_function(
    experiment: Experiment,
    loss_function: LossFcnCallableHint,
    grad_of_loss_function: None | LossFcnCallableHint,
) -> None:
    loss_fnc_fig, loss_ax = plot_loss_function(
        sales=experiment.sales_data,
        stocks=experiment.stocks_data,
        loss_function=loss_function,
        grad_function=grad_of_loss_function,
        true_mean=experiment.poisson_rate,
        x_axis_range_relative_to_mean=3.0,
        y_axis_limits=None,
    )
    save_or_show_image(
        fig=loss_fnc_fig,
        figure_file_path=output_path / f"loss_function_for_exp_{experiment.name}.png",
        overwrite_existing=True,
    )


def run_study(
    name: str,
    loss_function: LossFcnCallableHint,
    grad_of_loss_function: None | LossFcnCallableHint,
    sample_size: int,
    poisson_rate: float,
    min_max_stock: tuple[int, int],
    torch_random_seed: int,
    number_of_epochs: int,
    learning_rate: float,
    print_current_values: bool = False,
) -> None:
    current_experiment = Experiment.create_experiment(
        name=name,
        sample_size=sample_size,
        poisson_rate=poisson_rate,
        min_max_stock=min_max_stock,
        torch_random_seed=torch_random_seed,
    )

    evaluate_current_experiment(experiment=current_experiment)
    evaluate_loss_function(
        experiment=current_experiment,
        loss_function=loss_function,
        grad_of_loss_function=grad_of_loss_function,
    )

    best_mean_prediction = run_training(
        loss_function=loss_function,
        experiment=current_experiment,
        n_epochs=number_of_epochs,
        learning_rate=learning_rate,
        print_current_values=print_current_values,
    )

    print(f"Best mean prediction for experiment {current_experiment.name}: {best_mean_prediction}")


def main() -> None:
    run_study(
        name="ConradLoss_DefaultSetup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=2.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_OptimizedDefaultSetup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=2.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.005,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_1_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=1.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.001,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_2p5_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=2.5,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.005,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_3_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=3.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.005,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_3p5_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=3.5,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_4_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=4.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_8_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=8.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_10_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=10.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_12_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=12.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_16_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=16.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_20_Setup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=20.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )

    run_study(
        name="ConradLoss_Mean_20_OptimizedSetup",
        loss_function=conrad_negative_log_likelihood,
        grad_of_loss_function=grad_of_conrad_negative_log_likelihood,
        sample_size=default_n_data,
        poisson_rate=20.0,
        min_max_stock=(default_min_stock, default_max_stock),
        torch_random_seed=default_torch_seed,
        number_of_epochs=10 * default_epochs,
        learning_rate=0.01,
        print_current_values=False,
    )


if __name__ == "__main__":
    main()
