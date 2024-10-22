#!/usr/bin/env python

from collections.abc import Callable

import torch

__all__ = [
    "poisson_cdf",
    "LossFcnCallableHint",
    "conrad_negative_log_likelihood",
    "grad_of_conrad_negative_log_likelihood",
]


LossFcnCallableHint = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def poisson_cdf(
    mean: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the element-wise poisson cdf for the provided `mean` and `k` tensors.
    The arguments `mean` and `k` must be tensors of the same size.

    As torch does not provide a poisson CDF implementation, the CDF is calculated using the `torch.special.gammaincc`
    function, as suggested in https://github.com/pytorch/pytorch/issues/97156.
    The `torch.special.gammaincc` function reproduces the poisson CDF results well,
    except for the edge case of negative values of `k` in combination with `mean` == 0.0, where it will return NaN.
    This edge case is avoided by adding an epsilon value to `mean` values with `mean` == 0.

    :param mean: Tensor of mean values to consider. Tensor must fulfill `mean_i >= 0` for all elements i.
    :param k:
    :return: Tensor of element-wise poisson cdf
    """
    if not torch.all(torch.ge(mean, 0.0)):
        raise ValueError(
            f"All elements of mean must be greater than zero, but {torch.sum(torch.lt(mean, 0.0))} are not!"
        )

    _epsilon: float = 1e-10
    _k: torch.Tensor = torch.maximum(k, -1 * torch.ones_like(k))
    _m: torch.Tensor = torch.maximum(mean, torch.ones_like(mean) * _epsilon)
    return torch.special.gammaincc(torch.floor(_k + 1), _m)


def get_conrad_lower_bound_on_mean(
    sales_data: torch.Tensor,
    no_stock_out_mask: torch.Tensor,
) -> float:
    r"""
    Calculates the lower bound on the mean values from the sales and stocks data
    as provided in S. A. Conrad (1976):
    && m_{\text{min}} > \sum_{i=0}^{r} x_{i} / r $$
    This bound is calculated from all sales for which no stock out occurred (sales > stocks).

    If all data points are out-of-stock data points, the mean of the sales is returned.

    :param sales_data: Tensor of sales
    :param no_stock_out_mask: Boolean tensor which masks the entries of the sales tensor for which a stock out occurred.
    :return: Lower bound on mean values
    """

    # Catch case in which all data points are out of stock
    # -> Demand is much higher than sales -> returning the mean of the sales
    if not torch.any(no_stock_out_mask):
        return torch.mean(sales_data).item()

    _r: int = torch.sum(no_stock_out_mask).item()
    _sum_x: float = torch.sum(sales_data.mul(no_stock_out_mask)).item()

    return _sum_x / _r


def get_modified_mean_input(
    mean_input: torch.Tensor,
    lower_bound_on_mean: float,
) -> torch.Tensor:
    """
    Modifies the mean prediction input to avoid edge cases of the loss definition.
    Undesired edge cases occur for mean input values close to zero, as well as negative mean values.
    These cases can be avoided, as such mean values are below the lower bound on the mean values
    as provided in S. A. Conrad (1976).

    This function returns the absolute value of the input mean to avoid negative values
    AND
    adds a linear term to mean values below the lower bound `m_min`

    :param mean_input: Tensor of mean predictions.
    :param lower_bound_on_mean: Lower bound on the mean.
    :return: Tensor of modified mean predictions.
    """

    return torch.abs(mean_input) + 0.5 * torch.nn.functional.relu(lower_bound_on_mean - torch.abs(mean_input))


def conrad_negative_log_likelihood(
    mean_predictions: torch.Tensor,
    sales: torch.Tensor,
    stocks: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log likelihood calculated from the modified likelihood definition L(m)
    from S. A. Conrad: "Sales Data and the Estimation of Demand" (1976);
    L(m) is modified to consider non-constant stock values.

    :param mean_predictions: Tensor of mean predictions.
    :param sales: Tensor containing the sales data.
    :param stocks: Tensor containing the stock data.
    :return: Loss
    """

    no_stock_out: torch.Tensor = torch.gt(stocks, sales)
    stock_out: torch.Tensor = torch.logical_not(no_stock_out)

    no_stock_out_sales: torch.Tensor = sales.mul(no_stock_out)
    stock_out_sales: torch.Tensor = stocks.mul(stock_out)

    _mean_min: float = get_conrad_lower_bound_on_mean(sales_data=sales, no_stock_out_mask=no_stock_out)
    _m: torch.Tensor = get_modified_mean_input(mean_input=mean_predictions, lower_bound_on_mean=_mean_min)

    no_stock_out_part: torch.Tensor = torch.sum(no_stock_out_sales * torch.log(_m)) - torch.sum(no_stock_out * _m)
    stock_out_part: torch.Tensor = torch.sum(torch.log(1.0 - poisson_cdf(mean=stock_out * _m, k=stock_out_sales - 1)))

    return -1.0 * no_stock_out_part.add(stock_out_part)


def grad_of_conrad_negative_log_likelihood(
    mean_predictions: torch.Tensor,
    sales: torch.Tensor,
    stocks: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient of negative log likelihood calculated from the modified likelihood definition L(m)
    from S. A. Conrad: "Sales Data and the Estimation of Demand" (1976);
    L(m) is modified to consider non-constant stock values.

    :param mean_predictions: Tensor of mean predictions.
    :param sales: Tensor containing the sales data.
    :param stocks: Tensor containing the stock data.
    :return: Gradient of negative log likelihood
    """

    no_stock_out: torch.Tensor = torch.gt(stocks, sales)
    stock_out: torch.Tensor = torch.logical_not(no_stock_out)

    no_stock_out_sales: torch.Tensor = sales.mul(no_stock_out)
    stock_out_sales: torch.Tensor = stocks.mul(stock_out)

    _m_sign: torch.Tensor = torch.sign(mean_predictions)[0]
    _mean_min: float = get_conrad_lower_bound_on_mean(sales_data=sales, no_stock_out_mask=no_stock_out)
    _m: torch.Tensor = get_modified_mean_input(mean_input=mean_predictions, lower_bound_on_mean=_mean_min)

    no_stock_out_part: torch.Tensor = torch.sum(no_stock_out_sales.div(_m).add(-1.0 * no_stock_out))

    # stock_out_part: torch.Tensor = torch.sum(
    #     (torch.nan_to_num(poisson_cdf(mean=stock_out.mul(_m), k=stock_out_sales - 1)).add(
    #         torch.nan_to_num(poisson_cdf(mean=stock_out.mul(_m), k=stock_out_sales - 2)),
    #         alpha=-1.0)).div(
    #         1.0 - torch.nan_to_num(poisson_cdf(mean=stock_out.mul(_m), k=stock_out_sales - 1)))
    # )

    stock_out_part: torch.Tensor = torch.sum(
        (
            poisson_cdf(mean=stock_out.mul(_m), k=stock_out_sales - 1)
            - poisson_cdf(mean=stock_out.mul(_m), k=stock_out_sales - 2)
        ).div(1.0 - poisson_cdf(mean=stock_out.mul(_m), k=stock_out_sales - 1))
    )

    res: torch.Tensor = -1.0 * _m_sign * no_stock_out_part.add(stock_out_part)

    return res
