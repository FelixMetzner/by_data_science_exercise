#!/usr/bin/env python

import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scipy_stats
import torch

from by_data_science_exercise.loss_definitions import LossFcnCallableHint

__all__ = [
    "plot_distributions",
    "plot_loss_function",
    "plot_loss",
    "save_or_show_image",
]


@dataclass(frozen=True)
class DistMean:
    mean: float
    lower_bound: None | float
    upper_bound: None | float

    @staticmethod
    def calculate_from(
        data: np.ndarray,
        calculate_uncertainties: bool,
    ) -> "DistMean":
        _m, bounds = DistMean._calculate_mean(data=data, calculate_uncertainties=calculate_uncertainties)
        return DistMean(
            mean=_m,
            lower_bound=None if bounds is None else bounds[0],
            upper_bound=None if bounds is None else bounds[1],
        )

    @staticmethod
    def _calculate_mean(
        data: np.ndarray,
        calculate_uncertainties: bool,
    ) -> tuple[float, None | tuple[float, float]]:
        mean: float = float(np.mean(data))

        if not calculate_uncertainties:
            return mean, None

        n_events: int = len(data)
        k_sum: float = float(np.sum(data))
        quantile: float = (1 - 0.6827) / 2.0

        lower_bound: float = float(scipy_stats.gamma.ppf(quantile, k_sum, loc=0, scale=1) / n_events)
        upper_bound: float = float(scipy_stats.gamma.ppf(1.0 - quantile, k_sum + 1, loc=0, scale=1) / n_events)

        return mean, (lower_bound, upper_bound)


def plot_distributions(
    demand: torch.Tensor,
    sales: torch.Tensor,
    stocks: None | torch.Tensor = None,
    show_uncert: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    demand_array: np.ndarray = demand.numpy()
    sales_array: np.ndarray = sales.numpy()
    _arrays: list[np.ndarray] = [demand_array, sales_array]

    stocks_array: None | np.ndarray = None if stocks is None else stocks.numpy()
    if stocks_array is not None:
        _arrays.append(stocks_array)

    max_value: int = int(np.ceil(np.max(_arrays)))
    bin_edges: int | tuple[float, ...] = tuple(np.linspace(-0.5, max_value + 0.5, max_value + 2))

    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(6, 4), tight_layout=True)

    if stocks is not None:
        ax.hist(stocks_array, label="Stocks", bins=bin_edges, histtype="step", color="green")

    ax.hist(demand_array, label="Demand", bins=bin_edges, histtype="step", color="orange")
    ax.hist(sales_array, label="Sales", bins=bin_edges, histtype="step", color="blue")

    max_y = max([int(np.max(np.unique(_a, return_counts=True)[1])) for _a in _arrays])

    sales_mean: DistMean = DistMean.calculate_from(data=sales_array, calculate_uncertainties=show_uncert)
    demand_mean: DistMean = DistMean.calculate_from(data=demand_array, calculate_uncertainties=show_uncert)

    ax.vlines(
        x=sales_mean.mean,
        ymin=0,
        ymax=1.1 * max_y,
        label=f"Sales Mean ~{sales_mean.mean:.3f}",
        color="blue",
        linestyles=":",
    )
    ax.vlines(
        x=demand_mean.mean,
        ymin=0,
        ymax=1.1 * max_y,
        label=f"Demand Mean ~{demand_mean.mean:.3f}",
        color="orange",
        linestyles="--",
    )

    if show_uncert:
        ax.vlines(x=sales_mean.lower_bound, ymin=0, ymax=1.1 * max_y, color="blue", linestyles=":", linewidth=0.5)
        ax.vlines(x=sales_mean.upper_bound, ymin=0, ymax=1.1 * max_y, color="blue", linestyles=":", linewidth=0.5)
        ax.vlines(x=demand_mean.lower_bound, ymin=0, ymax=1.1 * max_y, color="orange", linestyles="--", linewidth=0.5)
        ax.vlines(x=demand_mean.upper_bound, ymin=0, ymax=1.1 * max_y, color="orange", linestyles="--", linewidth=0.5)

    ax.legend()
    return fig, ax


def plot_loss_function(
    sales: torch.Tensor,
    stocks: torch.Tensor,
    loss_function: LossFcnCallableHint,
    grad_function: LossFcnCallableHint | None,
    true_mean: float,
    x_axis_range_relative_to_mean: float = 3.0,
    y_axis_limits: None | tuple[float, float] = None,
) -> tuple[plt.Figure, plt.Axes]:
    x_vals_count: int = 10000

    x_vals: np.ndarray = np.linspace(
        start=-1.0 * x_axis_range_relative_to_mean * true_mean,
        stop=x_axis_range_relative_to_mean * true_mean,
        num=x_vals_count,
    )

    def _loss_fnc(m: torch.Tensor) -> torch.Tensor:
        return loss_function(m, sales, stocks)

    y_vals: np.ndarray = np.array([float(_loss_fnc(m=torch.ones_like(sales) * _x)) for _x in x_vals])

    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(6, 4), tight_layout=True)

    ax.plot(x_vals, y_vals, color="blue", label="loss")

    if grad_function is not None:

        def _grad_fnc(m: torch.Tensor) -> torch.Tensor:
            return grad_function(m, sales, stocks)

        dy_dx_vals: np.ndarray = np.array([float(_grad_fnc(torch.ones_like(sales) * _x)) for _x in x_vals])
        ax.plot(x_vals, dy_dx_vals, color="orange", label="gradient")

    ax.hlines(y=0.0, xmin=x_vals[0], xmax=x_vals[-1], color="red", linestyles="--", linewidth=0.8)

    if y_axis_limits is not None:
        ax.ylim(y_axis_limits)

    ax.set_xlabel("Mean values")

    ax.legend(frameon=False)

    return fig, ax


def plot_loss(
    loss_values: list[float],
    mean_values: None | list[float] = None,
) -> tuple[plt.Figure, plt.Axes]:
    epoch_vals: np.ndarray = np.arange(start=0, stop=len(loss_values)) + 1

    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(6, 4), tight_layout=True)

    ax.plot(epoch_vals, loss_values, color="blue", label="Loss")

    if mean_values is not None:
        mean_ax = ax.twinx()
        mean_ax.plot(epoch_vals, mean_values, color="orange", label="Current Mean")

        ax.set_ylabel("Loss", color="blue")
        mean_ax.set_ylabel("Current Mean", color="orange")

        ax.tick_params(axis="y", labelcolor="blue")
        mean_ax.tick_params(axis="y", labelcolor="orange")
    else:
        ax.set_ylabel("Loss")

    ax.set_xlabel("Epochs")

    return fig, ax


def save_or_show_image(
    fig: plt.Figure,
    figure_file_path: None | str | pathlib.Path = None,
    overwrite_existing: bool = False,
) -> None:
    if figure_file_path is None:
        fig.show()
    else:
        _figure_file_path: pathlib.Path = pathlib.Path(figure_file_path)

        if not _figure_file_path.parent.is_dir():
            raise NotADirectoryError(f"{str(_figure_file_path.parent.resolve())} is not a directory!")

        if _figure_file_path.exists() and not overwrite_existing:
            raise FileExistsError(f"The file {str(_figure_file_path.resolve())} already exists!")
        else:
            fig.savefig(_figure_file_path)

        plt.close(fig)
