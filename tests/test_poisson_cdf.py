#!/usr/bin/env python
import numpy
import numpy as np
import scipy.stats as scipy_stats
import torch

from by_data_science_exercise.loss_definitions import poisson_cdf


class TestPoissonCDF:
    def test_poisson_cdf_results(self) -> None:
        # Comparing own pytorch poisson_cdf implementation to the scipy version.

        mean_vals = np.linspace(0.0, 10.0, 100)
        k_vals = np.arange(-2, 20)
        k_tensor: torch.Tensor = torch.from_numpy(k_vals)

        for mean_val in mean_vals:
            expected_values: np.ndarray = scipy_stats.poisson.cdf(k=k_vals, mu=mean_val)
            own_values: torch.Tensor = poisson_cdf(mean=torch.ones_like(k_tensor) * mean_val, k=k_tensor)

            assert numpy.allclose(expected_values, own_values.numpy()), (
                mean_val,
                [
                    (k, e, o)
                    for k, e, o in zip(k_vals, expected_values, own_values.numpy(), strict=False)
                    if not np.isclose(e, o)
                ],
            )

            assert not np.any(np.isnan(expected_values)), np.sum(np.isnan(expected_values))
            assert not torch.any(torch.isnan(own_values)), torch.sum(torch.isnan(own_values))
