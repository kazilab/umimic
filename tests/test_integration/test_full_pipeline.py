"""Integration tests for the Experiment end-to-end pipeline."""

import numpy as np

from umimic.pipeline.experiment import Experiment


class TestFullPipeline:
    def test_generate_and_fit_end_to_end(self, invitro_quick_config):
        """Run generate_synthetic -> fit and verify result structure."""
        exp = Experiment(invitro_quick_config)
        dataset = exp.generate_synthetic()
        result = exp.fit(dataset)

        assert dataset.n_series > 0
        assert result.method == "mle"
        assert result.mle is not None
        assert result.mle.converged
        assert np.isfinite(result.mle.log_likelihood)
        assert len(result.mle.parameters) > 0

