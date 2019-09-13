import numpy as np
import pdb
import pytest
from lme.lme_forecast_verbose import LME
import lme.rutils as rutils


class TestLME:
    """Tests for `lme.lme_forecast_verbose` """

    @pytest.mark.parametrize("random_intercept", [[5, 4, 1, 1], [5, 1, 1, 1], [5, 4, 3, 2]])
    def test_random_intercept(self, random_intercept):
        dimensions = [5, 4, 3, 2]
        dct = {'intercept':[random_intercept[j] == dimensions[j] for j in range(len(dimensions))]}

        y = np.random.randn(np.prod(dimensions))
        model = LME(dimensions, 1, y, {}, {}, [], True, dct)
        Z = np.tile(rutils.kronecker(random_intercept[1:], dimensions, 1),(dimensions[0],1))
        model.buildZ()
        assert np.linalg.norm(Z-model.Z) == 0.0

    @pytest.mark.parametrize("indicator", [[5, 1, 1, 2], [1, 4, 1, 1], [5, 4, 3, 2], [1, 1, 1, 1]])
    def test_indicators(self, indicator):
        dimensions = [5, 4, 3, 2]
        dct = {'intercept': [indicator[j] == dimensions[j] for j in range(len(dimensions))]}

        y = np.random.randn(np.prod(dimensions))
        model = LME(dimensions, 0, y, {}, dct, [], False, {})
        Z = rutils.kronecker(indicator, dimensions, 0)
        x = np.random.randn(np.prod(indicator))

        assert (np.linalg.norm(model.X(x) - Z.dot(x)) < 1e-10) and \
             (np.linalg.norm(model.XT(y) - np.transpose(Z).dot(y)) < 1e-10)

    @pytest.mark.parametrize("cov_dim", [[1,1,1,5], [4,1,2,1], [4,3,2,5], [1,1,1,1]])
    def test_repeat_covariate(self, cov_dim):
        dimensions = [4, 3, 2, 5]
        N = np.prod(dimensions)
        X = np.ones((N,2))
        cov = np.random.randn(np.prod(cov_dim))
        cov_dim_bool = [cov_dim[i] == dimensions[i] for i in range(len(dimensions))]
        Z = rutils.kronecker(cov_dim, dimensions, 0)
        X[:, 1] = Z.dot(cov)
        beta_true = [1., -0.6]
        Y = X.dot(beta_true)
        model = LME(dimensions, 0, Y, {'cov1': (cov, cov_dim_bool)}, {}, ['cov1'],
                    True, {})
        beta = np.random.randn(2)
        assert np.linalg.norm(model.X(beta) - X.dot(beta)) < 1e-10
        y = np.random.randn(N)
        assert np.linalg.norm(model.XT(y) - np.transpose(X).dot(y)) < 1e-10

    def test_post_var_global(self):
        dimensions = [4, 3, 2, 2]
        N = np.prod(dimensions)
        X = np.random.randn(N,2)
        beta_true = [1., -0.6]
        Y_true = X.dot(beta_true)
        delta_true = .005
        Y = Y_true + np.random.randn(N)*np.sqrt(delta_true)
        model = LME(dimensions, 1, Y, {'cov1':(X[:,0], [True]*len(dimensions)),
                    'cov2': (X[:, 1], [True]*len(dimensions))}, {},
                    ['cov1', 'cov2'], False, {})
        model.optimize(inner_print_level=0)
        assert model.gamma_soln == 1e-8
        model.postVarGlobal()
        varmat1 = model.var_beta
        model._postVarGlobal()
        varmat2 = model.var_beta
        assert np.linalg.norm(varmat1 - varmat2) < 1e-10

    @pytest.mark.parametrize("random_effects", [[[9, 1, 2, 1], [9, 3, 1, 1]], [[9, 1, 1, 1], [9, 1, 2, 1]]])
    def test_draw_random_only(self, random_effects):
        np.random.seed(127)
        dimensions = [9, 3, 2, 2]
        N = np.prod(dimensions)
        Y_true = np.zeros(N)
        dct = {}
        for i, effect in enumerate(random_effects):
            Z = rutils.kronecker(effect, dimensions, 0)
            u = np.random.randn(np.prod(effect))
            Y_true += Z.dot(u)
            dct['intercept'+str(i)] = [effect[j] == dimensions[j] for j in range(len(dimensions))]
        delta_true = .005
        Y = Y_true + np.random.randn(N)*np.sqrt(delta_true)
        model = LME(dimensions, 1, Y, {},
                    {}, [], False, random_effects=dct)
        model.optimize(inner_print_level=0)
        model.postVarRandom()
        n_draws = 1000
        _, u_samples = model.draw(n_draws=n_draws)

        u1 = np.concatenate([u[:np.prod(random_effects[0][1:])] for u in model.u_soln])
        u1_sample_mean = np.mean(u_samples[0].reshape((-1, n_draws)),axis=1)
        assert np.linalg.norm(u1 - u1_sample_mean)/np.linalg.norm(u1) < .05
        u2 = np.concatenate([u[np.prod(random_effects[0][1:]):np.prod(random_effects[0][1:])
                               +np.prod(random_effects[1][1:])] for u in model.u_soln])
        u2_sample_mean = np.mean(u_samples[1].reshape((-1, n_draws)),axis=1)
        assert np.linalg.norm(u2 - u2_sample_mean)/np.linalg.norm(u2) < .05

        model.outputDraws()

        return

    @pytest.mark.parametrize("random_effects", [[[9, 1, 2, 1], [9, 3, 1, 1]], [[9, 1, 1, 1], [9, 1, 2, 1]], []])
    def test_draw(self, random_effects):
        np.random.seed(127)
        dimensions = [9, 3, 2, 2]
        N = np.prod(dimensions)
        X = np.ones((N,2))
        X[:, 1] = np.random.randn(N)
        beta_true = [1., -0.6]
        Y_true = X.dot(beta_true)
        dct = {}
        for i, effect in enumerate(random_effects):
            Z = rutils.kronecker(effect, dimensions, 0)
            u = np.random.randn(np.prod(effect))
            Y_true += Z.dot(u)
            dct['intercept'+str(i)] = [effect[j] == dimensions[j] for j in range(len(dimensions))]
        delta_true = .005
        Y = Y_true + np.random.randn(N)*np.sqrt(delta_true)
        model = LME(dimensions, 1, Y, {'cov':(X[:,1], [True]*len(dimensions))},
                    {}, ['cov'], True, random_effects=dct)
        model.optimize(inner_print_level=0)
        model.postVarGlobal()
        if len(random_effects) > 0:
            model.postVarRandom()
        n_draws = 1000
        beta_samples, u_samples = model.draw(n_draws=n_draws)
        beta_sample_mean = np.mean(beta_samples, axis=1)
        assert np.linalg.norm(beta_sample_mean - model.beta_soln)/np.linalg.norm(model.beta_soln) < .02

        if len(random_effects) > 0:
            u1 = np.concatenate([u[:np.prod(random_effects[0][1:])] for u in model.u_soln])
            u1_sample_mean = np.mean(u_samples[0].reshape((-1, n_draws)),axis=1)
            assert np.linalg.norm(u1 - u1_sample_mean)/np.linalg.norm(u1) < .05
            u2 = np.concatenate([u[np.prod(random_effects[0][1:]):np.prod(random_effects[0][1:])
                                   +np.prod(random_effects[1][1:])] for u in model.u_soln])
            u2_sample_mean = np.mean(u_samples[1].reshape((-1, n_draws)),axis=1)
            assert np.linalg.norm(u2 - u2_sample_mean)/np.linalg.norm(u2) < .05
        model.outputDraws()
