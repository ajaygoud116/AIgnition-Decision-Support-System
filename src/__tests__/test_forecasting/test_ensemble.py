import numpy as np
import pytest

from src.forecasting.ensemble import EnsembleForecaster
from src.models.common import QuantileValue


class TestEnsembleForecaster:
    def test_default_weights(self):
        ens = EnsembleForecaster()
        assert ens._lgb_weight == 0.5
        assert ens._sn_weight == 0.5

    def test_combine_returns_average_with_equal_weights(self):
        ens = EnsembleForecaster(lgb_weight=0.5, sn_weight=0.5)
        lgb = np.array([[100.0, 200.0, 300.0]])
        sn = np.array([[80.0, 180.0, 280.0]])
        result = ens.combine(lgb, sn)
        assert result[0, 0] == pytest.approx(90.0)
        assert result[0, 1] == pytest.approx(190.0)
        assert result[0, 2] == pytest.approx(290.0)

    def test_combine_with_skewed_weights(self):
        ens = EnsembleForecaster(lgb_weight=0.8, sn_weight=0.2)
        lgb = np.array([[100.0]])
        sn = np.array([[200.0]])
        result = ens.combine(lgb, sn)
        assert result[0, 0] == pytest.approx(120.0)

    def test_combine_1d_inputs(self):
        ens = EnsembleForecaster(lgb_weight=0.5, sn_weight=0.5)
        lgb = np.array([100.0, 200.0, 300.0])
        sn = np.array([80.0, 180.0, 280.0])
        result = ens.combine(lgb, sn)
        assert result[0, 0] == pytest.approx(90.0)

    def test_to_quantile_value(self):
        ens = EnsembleForecaster()
        pred = np.array([10.0, 20.0, 30.0])
        qv = ens.to_quantile_value(pred)
        assert isinstance(qv, QuantileValue)
        assert qv.p10 == 10.0
        assert qv.p50 == 20.0
        assert qv.p90 == 30.0
