import numpy as np
import pytest
import lme.rutils as rutils


class TestRUtils:
    """Test for functions in `lme.rutils` """

    @pytest.mark.parametrize("full_dims, dims", [([5, 4, 3, 2], [1, 4, 1, 2]),
                             ([5, 4, 3, 2], [1, 1, 1, 1]), ([5, 4, 3, 2], [5, 4, 3, 2]),
                             pytest.param([5, 4, 3, 2], [1, 3, 1, 1], marks=pytest.mark.xfail)])
    def test_repeat_kron(self, full_dims, dims):
        values = np.random.randn(np.prod(dims))

        Z = rutils.kronecker(dims, full_dims, 0)
        vals = rutils.repeat(values, dims, full_dims)
        y = np.random.randn(np.prod(full_dims))
        vals2 = rutils.repeatTranspose(y, dims, full_dims)

        assert (np.linalg.norm(Z.dot(values) - vals) < 1e-10
                and np.linalg.norm(np.transpose(Z).dot(y) - vals2) < 1e-10)
