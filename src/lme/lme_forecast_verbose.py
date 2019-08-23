import sys
path = '/Users/jizez/Dropbox (uwamath)/limetr.git/'
sys.path.insert(0, path)
from limetr import LimeTr
import numpy as np
import lme.utils as utils
import lme.rutils as rutils
import copy
import time

### TODO list
# - test draw()

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
            global_effects_names (list[str]):
                A list of covariates names that will be used as global effects.
            global_intercept (boolean):
                A boolean indicating whether to use an intercept in global effects.
            random_effects (dict of str: list[boolean]):
                A mapping that stores info of random effects. Key is the name of
                covariate or of intercept, and value is a boolean list specifying
                dimensions on which to impose random effects, e.g. if we want to
                know random slope of covariate id=0 per location-age,
                { 'haq': [True, True, False, False]},
                if we want to add a random intercept on location, then
                { 'haq': [True, True, False, False],
                  'intercept_loc': [True, False, False, False]}.
                Any name that does not appear in ``covariates`` will be interpreted
                as name for an intercept. Note that the first ``n_grouping_dims``
                of the boolean list must always be True for all random effects.
        """
        if any([d <= 1 for d in dimensions]):
            err_msg = 'Dimensions should all be > 1.'
            raise ValueError(err_msg)
        if n_grouping_dims < 0:
            err_msg = 'n_grouping_dims should be nonnegative.'
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

        def bool_to_size(b):
            assert len(b) == self.nd
            for i in range(self.nd):
                b[i] = max(1, int(b[i])*self.dimensions[i])

        self.covariates = []
        self.cov_name_to_id = {}
        i = 0
        for name, pair in covariates.items():
            #assert len(pair) == 2
            if len(pair) != 2:
                err_msg = 'input for ' + name + 'is not in correct form.'
                raise RuntimeError(err_msg)
            bool_to_size(pair[1])
            assert len(pair[0]) == np.prod(pair[1])
            self.covariates.append(pair)
            self.cov_name_to_id[name] = i
            i += 1

        #assert not (global_intercept and indicators != {})
        if global_intercept and indicators != {}:
            err_msg = 'cannot have both global intercept be True and indicators \
                       be non-empty.'
            raise RuntimeError(err_msg)
        self.global_ids = [self.cov_name_to_id[name] for name in global_effects_names]
        self.global_intercept = global_intercept
        self.k_beta = len(self.global_ids) + int(self.global_intercept)
        self.indicators = []
        self.indicator_name_to_id = {}
        i = 0
        for name, ind in indicators.items():
            bool_to_size(ind)
            self.k_beta += np.prod(ind)
            self.indicators.append(ind)
            self.indicator_name_to_id[name] = i
            i += 1
        self.ran_list = []
        for name, ran_eff in random_effects.items():
            #assert all(ran_eff[:self.n_grouping_dims])
            if not all(ran_eff[:self.n_grouping_dims]):
                err_msg = 'the first ' + str(self.n_grouping_dims) + ' must be \
                           True for random effects.'
                raise RuntimeError(err_msg)
            bool_to_size(ran_eff)
            if name in self.cov_name_to_id:
                self.ran_list.append((self.cov_name_to_id[name], ran_eff))
            else:
                self.ran_list.append((None, ran_eff))

        self.add_re = True
        if self.ran_list == []:
            self.add_re = False
        return

    def X(self, beta, naive=False):
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

    def buildZ(self):
        Z = []
        self.k_gamma = 0
        for ran in self.ran_list:
            id, dims = ran
            values = []
            if id == None:
                values = np.ones(self.N)
            else:
                values = rutils.repeat(self.covariates[id][0],self.covariates[id][1], self.dimensions)
            self.k_gamma += np.prod(dims[self.n_grouping_dims:])
            Z.append(values.reshape((-1,1))*np.tile(rutils.kronecker(dims[self.n_grouping_dims:], self.dimensions, self.n_grouping_dims),(self.n_groups,1)))

        self.Z = np.hstack(Z)
        #print(self.Z.shape)

    def optimize(self, var=None, S=None, uprior=None, trim_percentage=0.0,
                 share_obs_std=True, fit_fixed=True,inner_print_level=5,
                 inner_max_iter=100, inner_tol=1e-5,outer_verbose=False,
                 outer_max_iter=1, outer_step_size=1,
                 outer_tol=1e-6):
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
            uprior (numpy.ndarray | None, optional):
                Two-dimensional array where one row stores lower bounds for
                each variable and the other row stores upper bounds.
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

        """
        self.S = S
        self.share_obs_std = share_obs_std
        self.buildZ()
        k = self.k_beta + self.k_gamma
        if S is None:
            if share_obs_std:
                k += 1
            else:
                k += len(self.grouping)
        print('n_groups', self.n_groups)
        print('k_beta', self.k_beta)
        print('k_gamma', self.k_gamma)
        print('total number of fixed effects variables',k)

        C = []
        start = self.k_beta
        for ran in self.ran_list:
            _, dims = ran
            m = np.prod(dims[self.n_grouping_dims:])
            c = np.zeros((m-1,k))
            for i in range(m-1):
                c[i,start+i] = 1
                c[i,start+i+1] = -1
            C.append(c)
            start += m
        if len(C) > 0:
            self.constraints = np.vstack(C)
            assert self.constraints.shape[1] == k
        else:
            self.constraints = []

        C = None
        #if self.constraints != []:
        if len(self.constraints) > 0:
            C = lambda var: self.constraints.dot(var)

        JC = None
        #if self.constraints != []:
        if len(self.constraints) > 0:
            JC = lambda var: self.constraints

        c = None
        #if self.constraints != []:
        if len(self.constraints) > 0:
            c = np.zeros((2,self.constraints.shape[0]))

        up = []
        if uprior is None:
            up = np.array([
                [-np.inf]*self.k_beta + [1e-8]*self.k_gamma +\
                    [1e-7]*(k-self.k_beta-self.k_gamma),
                [np.inf]*k
                ])
        else:
            up = uprior

        x0 = np.ones(k)*.1
        if var is not None:
            assert len(var) == k
            x0 = var

        if var is None or fit_fixed or self.add_re == False:
            uprior_fixed = copy.deepcopy(up)
            uprior_fixed[:,self.k_beta:self.k_beta+self.k_gamma] = 1e-8
            model_fixed = LimeTr(self.grouping, self.k_beta, self.k_gamma, self.Y, self.X, self.XT, \
                                 self.Z, S=S, C=C, JC=JC, c=c, inlier_percentage=1.-trim_percentage,\
                                 share_obs_std=share_obs_std, uprior=uprior_fixed)
            #print('fit with gamma fixed...')
            t0 = time.time()
            model_fixed.optimize(x0=x0,print_level=inner_print_level,max_iter=inner_max_iter)
            #print('finished...elapsed',time.time()-t0)

            x0 = model_fixed.soln
            self.beta_fixed = model_fixed.beta
            if self.add_re == False:
                self.beta_soln = self.beta_fixed
                self.delta_soln = model_fixed.delta
                self.w_soln = model_fixed.w
                self.info = model_fixed.info['status_msg']
                self.yfit_no_random = model_fixed.X(model_fixed.beta)
                return

        model = LimeTr(self.grouping, self.k_beta, self.k_gamma, self.Y, self.X, self.XT, \
                       self.Z, S=S, C=C, JC=JC, c=c, inlier_percentage=1-trim_percentage,\
                       share_obs_std=share_obs_std, uprior=up)
        model.fitModel(x0=x0,
                     inner_print_level=inner_print_level,
                     inner_max_iter=inner_max_iter,
                     inner_tol=inner_tol,
                     outer_verbose=outer_verbose,
                     outer_max_iter=outer_max_iter,
                     outer_step_size=outer_step_size,
                     outer_tol=outer_tol)
        self.beta_soln = model.beta
        self.gamma_soln = model.gamma
        self.delta_soln = model.delta
        self.info = model.info
        self.w_soln = model.w
        self.u_soln = model.estimateRE()
        self.info = model.info['status_msg']

        self.yfit_no_random = model.X(model.beta)

        self.yfit = []
        Z_split = np.split(self.Z,self.n_groups)
        yfit_no_random_split = np.split(self.yfit_no_random, self.n_groups)

        for i in range(self.n_groups):
            self.yfit.append(yfit_no_random_split[i] + Z_split[i].dot(self.u_soln[i]))
        self.yfit = np.concatenate(self.yfit)
        self.model = model

    def postVarRandom(self):
        """
        y_k = X_k beta + Z_k u_k + epsilon, u has var_mat D, epsilon has var_mat R

        Var(u_k) = inv(inv(D) + Z_k'inv(R)Z_k)
        """
        assert len(self.ran_list) > 0
        Z_split = np.split(self.Z,self.n_groups)
        self.var_u = []
        S2 = []
        if self.S == None:
            if self.share_obs_std == True:
                S2 = np.ones(self.N)*self.delta_soln
            else:
                S2 = np.repeat(self.delta_soln, self.grouping)
        else:
            S2 = self.S**2
        S2_split = np.split(S2, self.n_groups)
        for i in range(self.n_groups):
            self.var_u.append(np.linalg.inv(np.diag(1./self.gamma_soln) +
                             (np.transpose(Z_split[i])/S2_split[i]).dot(Z_split[i])))


    def postVarGlobal(self):
        """
        y_k = X_k beta + Z_k u_k + epsilon, u has var_mat D, epsilon has var_mat R

        Var(beta) = inv( sum_k X_k' inv(Z_k*D*Z_k' + R) X_k )

        """
        assert self.k_beta > 0
        Z_split = np.split(self.Z,self.n_groups)

        self.var_beta = np.zeros((self.k_beta, self.k_beta))

        X = np.zeros((self.N, self.k_beta))
        start = 0

        if self.global_intercept == True:
            X[:,start] = np.ones(self.N)
            start += 1

        for i in range(len(self.global_ids)):
            ind = self.global_ids[i]
            values, dims = self.covariates[ind]
            assert values.shape[0] == np.prod(dims)
            X[:,start] = rutils.repeat(values, dims, self.dimensions)
            start += 1

        for indicator in self.indicators:
            X[:,start:start + np.prod(indicator)] = rutils.kronecker(indicator, self.dimensions, 0)
            start += np.prod(indicator)

        X_split = np.split(X, self.n_groups)

        S2 = []
        if self.S == None:
            if self.share_obs_std == True:
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

    def sampleGlobalWithLimeTr(self, sample_size=100, max_iter=300):
        beta_samples,gamma_samples = LimeTr.sampleSoln(self.model, sample_size=sample_size, max_iter=max_iter)
        return beta_samples, gamma_samples

    def draw(self, n_draws=10):
        beta_samples = np.transpose(np.random.multivariate_normal(self.beta_soln, self.var_beta, n_draws))
        u_samples = [[] for _ in range(len(self.ran_list))]
        for i in range(self.n_groups):
            # sample all random effects u in global group i
            samples = np.random.multivariate_normal(self.u_soln[i], self.var_u[i], n_draws)
            start = 0
            for j in range(len(self.ran_list)):
                ran_eff = self.ran_list[j]
                dims = ran_eff[1]
                # extract u related to random effect j
                u_samples[j].append(samples[:,start:start + np.prod(dims[self.n_grouping_dims:])].reshape((n_draws,-1)))
                start += np.prod(dims[self.n_grouping_dims:])
        for i in range(len(u_samples)):
            # each u_sample is a matrix of dimension n_draws-by-n_groups specific
            # to that random effect (>= number of global groups)
            u_samples[i] = np.transpose(np.hstack(u_samples[i]))
        for i in range(len(self.ran_list)):
            ran_eff = self.ran_list[i]
            _, dims = ran_eff
            u_samples[i] = u_samples[i].reshape(tuple(dims +[n_draws])).squeeze()
        # each u_samples[i] have different shapes
        return beta_samples, u_samples
