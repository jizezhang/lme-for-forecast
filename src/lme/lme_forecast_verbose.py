import numpy as np
import lme.utils as utils
import lme.rutils as rutils
import copy
import time
from collections import namedtuple
# import sys
# path = '/Users/jizez/Dropbox (uwamath)/limetr.git/src'
# sys.path.insert(0, path)
from limetr import LimeTr
from limetr.utils import VarMat


class LME:

    def __init__(self, dimensions, n_grouping_dims, y, covariates,
                 indicators, global_effects_names,
                 global_intercept, random_effects):

        """
        Create a linear mixed effects model (LME) object

        .. math::
            {\bf y} = \beta_0 + X {\bf \beta_c} + K{\bf \beta_I} + Z \{bf u} + \epsilon

        where
            :math:`\beta_0` is a global intercept.
            :math:`X` is a matrix for global effects covariates.
            :math:`\beta_c` is a vector of covariate coefficients.
            :math:`K` is a matrix representing indicator functions
                (more details later).
            :math:`\beta_I` is a vector of indicator coefficients.
            :math:`Z` is a matrix of random effects terms (including
                      random effects covariates and random intercepts).
            :math:`u` is a vector of random effects coefficients. :math:`u` is
                      assumed to follow :math:`N(0, \Tau)` where :math:`\Tau` is
                      a diagonal matrix.
            :math:`\epsilon` is a vector of measurement noise terms.
                :math:`\epsilon` is assumed to follow :math:`N(0, \Sigma)`
                where :math:`\Sigma` is a diagonal matrix.

        To summarize, the fixed effects variables are
        .. math::
            `\beta_0, \beta_c, \beta_I, \text{diag}(\Tau), \text{diag}(\Sigma)`
        and the random effects variable is :math:`u`.

        This implementation of LME is designed for input data, including
        measurements data, covariates data, etc., that are organized by a
        certain set of dimensions, e.g. location-age-sex-year. The aforementioned
        indicator functions are indicator over sets defined by a single dimension
        or cartesian product of dimensions. For instance, the sets can be defined
        by locations, i.e. data entries from the same location belong to the one
        set, and the number of sets will be equal to the number of locations.
        The length of :math:`\beta_I` will be equal to the number of sets.
        The sets can also be defined by location-age, i.e. data entries from
        the same location and the same age group belong to one set. In this case
        the number of sets is equal to the number of locations times the number
        of age groups.

        Because of the special structure of input data, the matrices
        :math:`X, K` and their Jacobians are not formed explicitly in the
        implementation, in order to reduce memory usage and to speed up
        computation. :math:`Z` matrix is also not formed in its full dimensions.
        Instead we only pass in its diagonal blocks, since it should have a
        block diagonal structure in general.

        The underlying core optimization engine for this program is `LimeTr`,
        which can handle a more general class of linear mixed effects models with
        built-in robust outlier detection methods.

        Args:
            dimensions (list[int]):
                a list specifying dimensions of data. The value of each component
                specifies the size of that dimension, e.g. if the data has
                dimension location-age-sex-year, the list would
                be [n_loc, n_age, n_sex, n_year].
            n_grouping_dims (int):
                an int specifying the number of dimensions on which grouping occurs,
                e.g. if ``dimensions`` is [n_loc,n_age,n_sex,n_year] and
                `n_grouping_dims` is 2, then the data will be grouped by location
                and age, and the :math:`Z` matrix corresponding to random effects
                will be a block diagonal matrix with ``n_loc`` times ``n_age``
                number of blocks.
            y (numpy.ndarray):
                One-dimensional array storing ``y`` (dependent variable) values,
                and the values must be ordered according to the ``dimensions``,
                e.g. if ``dimensions`` is [n_loc,n_age,n_sex,n_year],
                then ``y`` must be ordered by location, age, sex, year.
            covariates (dict of str: (numpy.ndarray, list[boolean])):
                A mapping that stores info of covariates. Key is the name of
                the covariate, and value is a tuple where the first component is
                a one-dimensional array that stores values of that covariate, and
                the second component specifies the dimensions of covriate
                through a boolean list, e.g. for HAQ, the tuple would be
                (values_of_haq, [True, False, False, True])
                since HAQ does not depend on age and sex.
                The dictionary would look like
                {'haq': (values_of_haq, [True, False, False, True)}
            indicators(dict of str: list[boolean]):
                A mapping for indicators.
                Key is the name of indicator, value is a boolean list that specifies
                dimensions on which to use indicators, e.g. if age-sex indicator,
                dictionary would be
                {'ind_age-sex':[False, True, True, False]}, and
                if there is an age indicator and a sex indicator, it would be
                {'ind_age':[False, True, False, False],
                 'ind_sex':[False, False, True, False]}
            global_effects_names (dict of str: list[float]):
                A mapping that stores covariates names that will be used as global effects
                and corresponding bounds on covariate multiplier values.
            global_intercept (boolean):
                A boolean indicating whether to use an intercept in global effects.
            random_effects (dict of str: list[boolean]):
                A mapping that stores info of random effects. Key is the name of
                covariate or of intercept, and value is a boolean list specifying
                dimensions on which to impose random effects, e.g. if we want to
                know random slope of covariate 'haq' per location-age,
                { 'haq': [True, True, False, False]},
                if we want to add a random intercept on location, then
                { 'haq': [True, True, False, False],
                  'intercept_loc': [True, False, False, False]}.
                Any name that does not appear in ``covariates`` will be interpreted
                as name for an intercept. Note that the first ``n_grouping_dims``
                of the boolean list must always be True for all random effects.

        Attributes:
            beta_soln (numpy.ndarray): solution for global effects coefficients
            u_soln (list(numpy.ndarray)): solution for random effects coefficients
            yfit (numpy.ndarray): fitted y (dependent variable) values
            solve_status_msg (str): convergence info

        Raises:
            ValueError:
                If any dimension in ``dimensions`` is less than or equal to 1.
            ValueError:
                If `n_grouping_dims` is negative.
            RuntimeError:
                If value or boolean list for any covariate pair in ``covariates`` is missing.
            RuntimeError:
                If ``global_intercept`` is True and ``indicators`` is nonempty.
            RuntimeError:
                If the first ``n_grouping_dims`` of a random effect is not all True.
        """
        if any([d <= 1 for d in dimensions]):
            err_msg = 'Dimensions should all be > 1.'
            raise ValueError(err_msg)
        if n_grouping_dims < 0:
            err_msg = 'n_grouping_dims should be non-negative.'
            raise ValueError(err_msg)
        self.dimensions = dimensions
        self.n_grouping_dims = n_grouping_dims
        self.n_groups = int(np.prod(dimensions[:n_grouping_dims]))
        self.grouping = [np.prod(dimensions[n_grouping_dims:])]*self.n_groups
        self.Y = y
        self.N = self.Y.shape[0]
        self.nd = len(dimensions)
        assert self.N == np.prod(self.dimensions)
        assert self.N == sum(self.grouping)

        # def bool_to_size(b):
        #     assert len(b) == self.nd
        #     for i in range(self.nd):
        #         b[i] = max(1, int(b[i])*self.dimensions[i])

        def bool_to_size(b):
            assert len(b) == self.nd
            dims = [1 for _ in range(self.nd)]
            for i in range(self.nd):
                dims[i] = max(1, int(b[i])*self.dimensions[i])
            return dims

        self.covariates = []
        self.cov_name_to_id = {}
        i = 0
        for name, pair in covariates.items():
            if len(pair) != 2:
                err_msg = 'input for ' + name + 'is not in correct form.'
                raise RuntimeError(err_msg)
            dims = bool_to_size(pair[1])
            assert len(pair[0]) == np.prod(dims)
            self.covariates.append((pair[0], dims))
            self.cov_name_to_id[name] = i
            i += 1

        if global_intercept and indicators != {}:
            err_msg = 'cannot have both global intercept be True and indicators \
                       be non-empty.'
            raise RuntimeError(err_msg)
        self.global_ids = [self.cov_name_to_id[name] for name, bounds in global_effects_names.items()]
        if len(self.global_ids) > 0:
            self.global_cov_bounds = np.transpose(np.array([bounds for name, bounds in global_effects_names.items()]))
            assert self.global_cov_bounds.shape[0] == 2
            assert all(self.global_cov_bounds[1, :] - self.global_cov_bounds[0, :] >= 0)
        else:
            self.global_cov_bounds = None
        self.global_intercept = global_intercept
        self.k_beta = len(self.global_ids) + int(self.global_intercept)
        self.indicators = []
        self.indicator_name_to_id = {}
        i = 0
        for name, ind in indicators.items():
            dims = bool_to_size(ind)
            self.k_beta += np.prod(dims)
            self.indicators.append(dims)
            self.indicator_name_to_id[name] = i
            i += 1

        self.beta_names = list(global_effects_names.keys()) + list(self.indicator_name_to_id.keys())
        if self.global_intercept:
            self.beta_names = ['global_intercept'] + self.beta_names

        self.ran_list = []
        for name, ran_eff in random_effects.items():
            if not all(ran_eff[:self.n_grouping_dims]):
                err_msg = name + ': the first ' + str(self.n_grouping_dims) + ' must be \
                           True for random effects.'
                raise RuntimeError(err_msg)
            dims = bool_to_size(ran_eff)
            if name in self.cov_name_to_id:
                self.ran_list.append((self.cov_name_to_id[name], dims))
            else:
                self.ran_list.append((None, dims))

        self.u_names = list(random_effects.keys())

        return

    def X(self, beta):
        assert len(beta) == self.k_beta
        y = np.zeros(self.N)
        start = 0
        if self.global_intercept:
            y += beta[start]
            start += 1
        for i in range(len(self.global_ids)):
            ind = self.global_ids[i]
            values, dims = self.covariates[ind]
            assert values.shape[0] == np.prod(dims)
            y += rutils.repeat(values, dims, self.dimensions)*beta[start+i]
        start += len(self.global_ids)
        for indicator in self.indicators:
            y += rutils.repeat(beta[start:start + np.prod(indicator)],indicator, self.dimensions)
            start += np.prod(indicator)
        return y

    def XT(self, y):
        assert len(y) == self.N
        start = 0
        val = []
        if self.global_intercept:
            val.append(np.sum(y))
        for i in self.global_ids:
            values, dims = self.covariates[i]
            val.append(np.dot(values, rutils.repeatTranspose(y,dims, self.dimensions)))
        for indicator in self.indicators:
            val.extend(rutils.repeatTranspose(y, indicator, self.dimensions))
        assert len(val) == self.k_beta
        return np.array(val)

    def buildZ(self, normalize=False):
        Z = []
        self.k_gamma = 0
        for ran in self.ran_list:
            id, dims = ran
            values = []
            if id is None:
                values = np.ones(self.N)
            else:
                values = rutils.repeat(self.covariates[id][0],self.covariates[id][1], self.dimensions)
            self.k_gamma += np.prod(dims[self.n_grouping_dims:])
            Z.append(values.reshape((-1, 1)) *
                     np.tile(rutils.kronecker(dims[self.n_grouping_dims:], self.dimensions, self.n_grouping_dims),
                             (self.n_groups, 1)))
        if self.k_gamma > 0:
            self.Z = np.hstack(Z)
            col_norm = np.linalg.norm(self.Z, axis=0)
            if normalize:
                print('normalizing Z ...')
                print(col_norm)
                self.Z = self.Z/col_norm
            return col_norm
        else:
            self.Z = np.zeros((self.N, 1))
            return 0.0

    def optimize(self, var=None, S=None, trim_percentage=0.0,
                 share_obs_std=True, fit_fixed=True, inner_print_level=5,
                 inner_max_iter=100, inner_tol=1e-5, inner_verbose=True,
                 inner_acceptable_tol=1e-4, inner_nlp_scaling_min_value=1e-8,
                 outer_verbose=False, outer_max_iter=1, outer_step_size=1,
                 outer_tol=1e-6, normalize_Z=False, random_seed=0):
        """
        Run optimization routine via LimeTr.

        Args:
            var (numpy.ndarray | None, optional):
                One-dimensional array that gives initialization for variables.
                If None, the program will first run without random effects
                to obtain a starting point.
            S (numpy.ndarray | None, optional):
                One-dimensional numpy array that gives standard deviation for
                each measurement. The size of S should be the same as that of
                measurements vector. If None standard deviation of measurement
                will be treated as variables and optimized.
            trim_percentage (float | 0.0, optional):
                A float that gives percentage of datapoints to trim.
                Default is 0, i.e. no trimming.
            share_obs_std (boolean | True, optional):
                A boolean that indicates whether the model should assume data
                across studies share the same measurement standard deviation.
            fit_fixed (boolean | True, optional):
                A boolean that indicates whether to run a fit without random
                effects first in order to obtain a good starting point.
            inner_print_level (int | 5, optional):
                ``print_level`` for Ipopt.
            inner_max_iter (int | 100, optional):
                Maximum number of iterations for inner optimization.
            inner_tol (float | 1e-5, optional):
                Tolerance level for inner optimization.
            inner_verbose (boolean | True, optional):
                Verbose option for inner optimization.
            inner_acceptable_tol (float | 1e-4, optional):
                Acceptable tolerance level for inner optimization.
            inner_nlp_scaling_min_value (float | 1e-8, optional):
                Min scaling for objective function.
            outer_verbose (boolean | False, optional):
                Verbose option for outer optimization.
            outer_max_iter (int | 1, optional):
                Maximum number of iterations for outer optimization. When there
                is no trimming, outer optimization is not needed, so the default
                is set to be 1.
            outer_step_size (float |1.0, optional):
                Step size for outer optimization. Used in trimming.
            outer_tol (float | 1e-6, optional):
                Tolerance level for outer optimization.
            normalize_Z (bool | False, optional):
                Whether to normalize Z matrix before optimization.
            random_seed (int | 0, optional):
                random seed for choosing an initial point for optimization. If equals 0
                the initial point is chosen to be a vector of 0.01.
        """
        self.S = S
        self.share_obs_std = share_obs_std
        Z_norm = self.buildZ(normalize_Z)
        k = self.k_beta + self.k_gamma
        if S is None:
            if share_obs_std:
                k += 1
            else:
                k += len(self.grouping)
        print('n_groups', self.n_groups)
        print('k_beta', self.k_beta)
        print('k_gamma', self.k_gamma)
        print('total number of fixed effects variables', k)

        if self.k_gamma == 0:
            self.add_re = False
            self.k_gamma = 1
            k += 1
        else:
            self.add_re = True

        C = []
        start = self.k_beta
        for ran in self.ran_list:
            _, dims = ran
            m = np.prod(dims[self.n_grouping_dims:])
            c = np.zeros((m-1,k))
            for i in range(m-1):
                c[i, start+i] = 1
                c[i, start+i+1] = -1
            C.append(c)
            start += m
        if len(C) > 0:
            self.constraints = np.vstack(C)
            assert self.constraints.shape[1] == k
        else:
            self.constraints = []

        C = None
        if len(self.constraints) > 0:
            C = lambda var: self.constraints.dot(var)

        JC = None
        if len(self.constraints) > 0:
            JC = lambda var: self.constraints

        c = None
        if len(self.constraints) > 0:
            c = np.zeros((2, self.constraints.shape[0]))

        self.uprior = np.array([
                [-np.inf]*self.k_beta + [1e-7]*self.k_gamma + \
                    [1e-7]*(k-self.k_beta-self.k_gamma),
                [np.inf]*k
                ])

        if self.global_cov_bounds is not None:
            if self.global_intercept:
                self.uprior[:, 1:len(self.global_ids) + 1] = self.global_cov_bounds
            else:
                self.uprior[:, :len(self.global_ids)] = self.global_cov_bounds

        x0 = np.ones(k)*.01
        if random_seed != 0:
            np.random.seed(random_seed)
            x0 = np.random.randn(k)*.01
        if var is not None:
            if self.add_re is True:
                assert len(var) == k
                x0 = var
            else:
                assert len(var) == self.k_beta
                x0 = np.append(var, [1e-8])
                assert len(x0) == k

        if var is None or fit_fixed or self.add_re is False:
            uprior_fixed = copy.deepcopy(self.uprior)
            uprior_fixed[:, self.k_beta:self.k_beta+self.k_gamma] = 1e-8
            model_fixed = LimeTr(self.grouping, int(self.k_beta), int(self.k_gamma), self.Y, self.X, self.XT,
                                 self.Z, S=S, C=C, JC=JC, c=c, inlier_percentage=1.-trim_percentage,
                                 share_obs_std=share_obs_std, uprior=uprior_fixed)
            model_fixed.optimize(x0=x0, print_level=inner_print_level, max_iter=inner_max_iter,
                                 tol=inner_tol, acceptable_tol=inner_acceptable_tol,
                                 nlp_scaling_min_value=inner_nlp_scaling_min_value)

            x0 = model_fixed.soln
            self.beta_fixed = model_fixed.beta
            if self.add_re is False:
                self.beta_soln = self.beta_fixed
                self.delta_soln = model_fixed.delta
                self.gamma_soln = model_fixed.gamma
                self.w_soln = model_fixed.w
                self.info = model_fixed.info['status_msg']
                self.yfit_no_random = model_fixed.X(model_fixed.beta)
                return

        model = LimeTr(self.grouping, int(self.k_beta), int(self.k_gamma), self.Y, self.X, self.XT,
                       self.Z, S=S, C=C, JC=JC, c=c, inlier_percentage=1-trim_percentage,
                       share_obs_std=share_obs_std, uprior=self.uprior)
        model.fitModel(x0=x0,
                       inner_print_level=inner_print_level,
                       inner_max_iter=inner_max_iter,
                       inner_acceptable_tol=inner_acceptable_tol,
                       inner_nlp_scaling_min_value=inner_nlp_scaling_min_value,
                       inner_tol=inner_tol,
                       outer_verbose=outer_verbose,
                       outer_max_iter=outer_max_iter,
                       outer_step_size=outer_step_size,
                       outer_tol=outer_tol)
        self.beta_soln = model.beta
        self.gamma_soln = model.gamma
        if normalize_Z:
            self.gamma_soln /= Z_norm**2
        self.delta_soln = model.delta
        self.info = model.info
        self.w_soln = model.w
        self.u_soln = model.estimateRE()
        self.solve_status = model.info['status']
        self.solve_status_msg = model.info['status_msg']

        self.yfit_no_random = model.X(model.beta)

        self.yfit = []
        Z_split = np.split(self.Z,self.n_groups)
        yfit_no_random_split = np.split(self.yfit_no_random, self.n_groups)

        for i in range(self.n_groups):
            self.yfit.append(yfit_no_random_split[i] + Z_split[i].dot(self.u_soln[i]))
        self.yfit = np.concatenate(self.yfit)
        self.model = model

        if inner_verbose == True and self.solve_status != 0:
            print(self.solve_status_msg)

    def postVarRandom(self):
        """
        y_k = X_k beta + Z_k u_k + epsilon, u has var_mat D, epsilon has var_mat R

        Var(u_k) = inv(inv(D) + Z_k'inv(R)Z_k)
        """
        assert len(self.ran_list) > 0
        Z_split = np.split(self.Z, self.n_groups)
        self.var_u = []
        S2 = []
        if self.S is None:
            if self.share_obs_std is True:
                S2 = np.ones(self.N)*self.delta_soln
            else:
                S2 = np.repeat(self.delta_soln, self.grouping)
        else:
            S2 = self.S**2
        S2_split = np.split(S2, self.n_groups)
        for i in range(self.n_groups):
            self.var_u.append(np.linalg.inv(np.diag(1./self.gamma_soln) +
                             (np.transpose(Z_split[i])/S2_split[i]).dot(Z_split[i])))


    def _postVarGlobal(self):
        """
        y_k = X_k beta + Z_k u_k + epsilon, u has var_mat D, epsilon has var_mat R

        Var(beta) = inv( sum_k X_k' inv(Z_k*D*Z_k' + R) X_k )

        """
        assert self.k_beta > 0
        Z_split = np.split(self.Z, self.n_groups)
        Z_split = np.split(np.zeros((self.N, self.k_gamma)), self.n_groups)

        self.var_beta = np.zeros((self.k_beta, self.k_beta))

        X = np.zeros((self.N, self.k_beta))
        start = 0

        if self.global_intercept is True:
            X[:, start] = np.ones(self.N)
            start += 1

        for i in range(len(self.global_ids)):
            ind = self.global_ids[i]
            values, dims = self.covariates[ind]
            assert values.shape[0] == np.prod(dims)
            X[:, start] = rutils.repeat(values, dims, self.dimensions)
            start += 1

        for indicator in self.indicators:
            X[:, start:start + np.prod(indicator)] = rutils.kronecker(indicator, self.dimensions, 0)
            start += np.prod(indicator)

        X_split = np.split(X, self.n_groups)

        S2 = []
        if self.S is None:
            if self.share_obs_std is True:
                S2 = np.ones(self.N)*self.delta_soln
            else:
                S2 = np.repeat(self.delta_soln, self.grouping)
        else:
            S2 = self.S**2
        S2_split = np.split(S2, self.n_groups)

        for i in range(self.n_groups):
            V = Z_split[i].dot(np.diag(self.gamma_soln)).dot(np.transpose(Z_split[i])) \
                + S2_split[i]*np.identity(self.grouping[i])
            self.var_beta += np.transpose(X_split[i]).dot(np.linalg.inv(V)).dot(X_split[i])
        self.var_beta = np.linalg.inv(self.var_beta)

    def postVarGlobal(self):
        assert self.k_beta > 0

        X = np.zeros((self.N, self.k_beta))
        start = 0

        if self.global_intercept is True:
            X[:,start] = np.ones(self.N)
            start += 1

        for i in range(len(self.global_ids)):
            ind = self.global_ids[i]
            values, dims = self.covariates[ind]
            assert values.shape[0] == np.prod(dims)
            X[:,start] = rutils.repeat(values, dims, self.dimensions)
            start += 1

        for indicator in self.indicators:
            X[:, start:start + np.prod(indicator)] = rutils.kronecker(indicator, self.dimensions, 0)
            start += np.prod(indicator)

        S2 = []
        if self.S is None:
            if self.share_obs_std is True:
                S2 = np.ones(self.N)*self.delta_soln
            else:
                S2 = np.repeat(self.delta_soln, self.grouping)
        else:
            S2 = self.S**2

        mat = VarMat(S2, np.zeros((self.N, self.k_gamma)), self.gamma_soln, self.grouping)
        self.var_beta = np.dot(np.transpose(X), mat.invDot(X))
        self.var_beta = np.linalg.inv(self.var_beta)

    def sampleGlobalWithLimeTr(self, sample_size=100, max_iter=300):
        beta_samples,gamma_samples = LimeTr.sampleSoln(self.model, sample_size=sample_size, max_iter=max_iter)
        return beta_samples, gamma_samples

    def _drawBeta(self, n_draws):
        if self.global_cov_bounds is None:
            return np.transpose(np.random.multivariate_normal(self.beta_soln, self.var_beta, n_draws))

        bounds = self.uprior[:, :self.k_beta]
        print(bounds)
        beta_samples = np.empty((self.k_beta, 0), float)
        while beta_samples.shape[1] < n_draws:
            samples = np.transpose(np.random.multivariate_normal(self.beta_soln, self.var_beta, n_draws))
            beta_samples = np.hstack((beta_samples,
                                      samples[:, np.all((samples - bounds[0, :].reshape((-1, 1)) >= 0) &
                                                        (samples - bounds[1, :].reshape((-1, 1)) <= 0), axis=0)]))
        return beta_samples[:, :n_draws]

    def draw(self, n_draws=10):
        """
        Draw samples of global effects coefficient beta and random intercepts u
        from their corresponding posterior distribution.

        Args:
            n_draws(int | 10): number of draws

        Returns:
            list(numpy.ndarray):
                Arrays for beta samples and u samples
        """
        beta_samples = []
        if self.k_beta > 0:
            beta_samples = self._drawBeta(n_draws)
        u_samples = [[] for _ in range(len(self.ran_list))]
        if self.add_re is True:
            for i in range(self.n_groups):
                # sample all random effects u in global group i
                samples = np.random.multivariate_normal(self.u_soln[i], self.var_u[i], n_draws)
                start = 0
                for j in range(len(self.ran_list)):
                    ran_eff = self.ran_list[j]
                    dims = ran_eff[1]
                    # extract u related to random effect j
                    u_samples[j].append(samples[:, start:start + np.prod(dims[self.n_grouping_dims:])].reshape((n_draws, -1)))
                    start += np.prod(dims[self.n_grouping_dims:])
            for i in range(len(u_samples)):
                # each u_sample is a matrix of dimension n_draws-by-n_groups specific
                # to that random effect (>= number of global groups)
                # and the matrix is then transposed
                u_samples[i] = np.transpose(np.hstack(u_samples[i]))
            for i in range(len(self.ran_list)):
                ran_eff = self.ran_list[i]
                _, dims = ran_eff
                u_samples[i] = u_samples[i].reshape(tuple(dims +[n_draws])).squeeze()
        # each u_samples[i] have different shapes
        return beta_samples, u_samples

    def outputDraws(self, n_draws=10, by_type=True, combine_cov=True):
        """
        Output draws as namedtuples. Each tuple has two fields, `array` and `name(s)`.
        `array` stores a numpy array of draws. There are three types of estimated
        values we need draws: global covariates, indicators, and random effects.
        For all three types, the dimension of `array` is equal to the original
        dimension of the estimate, plus one extra dimension whose size is the number of draws.
        e.g.
        for covariates, its original dimension is 1, since it is a scalar,
        thus the `array` dimension is n_draws;
        for indicator, age-sex as an example. its original dimension is n_age-by-n_sex,
        thus the `array` dimension is n_age-by-n_sex-by-n_draws;
        random_effects is similar to indicators.

        Args:
            n_draws (int | 10): number of draws
            by_type (boolean | True):
                whether to group samples according to the three types
            combine_cov (boolean | True):
                whether to combine covariates into one namedtuple

        Returns:
            list(namedtuple):
                List of namedtuples. Each tuple is length two. The first component
                is a numpy array of draws, and the second is the name of the estimate.
                If ``by_type`` is true, estimates are grouped by type.
                If ``combine_cov`` is true, global covariates are combined into one
                namedtuple.

        """
        beta_samples, u_samples = self.draw(n_draws)
        samples = []
        n_cov = int(self.global_intercept) + len(self.global_ids)
        for i in range(n_cov):
            samples.append(beta_samples[i,:])
        start = n_cov
        for dim in self.indicators:
            samples.append(beta_samples[start:start+np.prod(dim),:].reshape(tuple(dim+[n_draws])).squeeze())
            start += np.prod(dim)
        assert start == self.k_beta
        assert len(samples) == len(self.beta_names)
        assert len(u_samples) == len(self.u_names)

        samples_name_pairs = zip(samples + u_samples, self.beta_names + self.u_names)
        Draws = namedtuple('Draws', 'array, name')
        samples_name_pairs = [Draws(*pair) for pair in samples_name_pairs]

        if by_type is True:
            cov_samples = samples_name_pairs[:n_cov]
            indicator_samples = samples_name_pairs[n_cov:len(self.beta_names)]
            raneff_samples = samples_name_pairs[len(self.beta_names):]
            if combine_cov is True:
                if n_cov > 0:
                    CovDraws = namedtuple('CovDraws', 'array, names')
                    cov_samples = CovDraws(beta_samples[:n_cov,:], self.beta_names[:n_cov])
                else:
                    cov_samples = None
            return cov_samples, indicator_samples, raneff_samples

        return samples_name_pairs
