import sys
path = '/Users/jizez/Dropbox (uwamath)/limetr.git/'
sys.path.insert(0, path)
from limetr import LimeTr
import numpy as np
import utils
import rutils
import copy
import time

### TODO list
# - test draw()
#

class LME:

    def __init__(self, dimensions, n_grouping_dims, y, covariates,
                 indicators, global_effects_indices,
                 global_intercept, random_effects_list):

        """
        Parameters:
        ----------
        dimensions: list
            dimensions of the data
            the length of the list is the # of dimensions
            the value of each component specifies the size of that dimension
            e.g. if the data has dimension location-age-sex-year, the list would
            be [n_loc,n_age,n_sex,n_year].
        n_grouping_dims: int
            the number of dimensions on which grouping occurs.
            takes the first `n_grouping_dims` from `dimensions`
            e.g. if `dimensions` is [n_loc,n_age,n_sex,n_year] and
            `n_grouping_dims` is 1, then the data will be grouped by location,
            and the Z matrix corresponding to random effects will be a block
            diagonal matrix with `n_loc` number of blocks
        y: 1D numpy array
            y (dependent variable) values, must be ordered according to the `dimensions`
            e.g. if `dimensions` is [n_loc,n_age,n_sex,n_year],
            then y must be ordered by location, age, sex, year.
        covariates: list of tuples
            each element of the list is a tuple of (1D numpy.array, dimensions_of_cov)
            where the dimensions of covriate is specified through a boolean list,
            corresponding to `dimensions`.
            e.g. for HAQ, the tuple would be
            (values_of_haq, [True, False, False, True])
            since HAQ does not depend on age and sex.
        indicators: list of lists
            each element is a boolean list specifying dimensions on which to use indicator
            e.g. if age-sex indicator, the list would be
            [[False, True, True, False]]
            if there is an age indicator and a sex indicator, the list would be
            [[False, True, False, False], [False, False, True, False]]
        global_effects_indices: list
            indices for covariates that will be used as global effects
        global_intercept: boolean
            whether to use an intercept in global effects
        random_effects_list: list of tuples
            each element of the list is a tuple of
            (id_of_cov, dimensions_of_random_effects_excluding_grouping_dimensions)
            e.g. if we want to know random slope of covariate id=0 per location-age,
            (0,[True, True, False, False])
            if we want to add a random intercept, then the first component of the tuple
            should be None. e.g. for an intercept on location, pass in
            (None, [True, False, False, False])
            Note that the first `n_grouping_dims` of the boolean list must always
            be True for all random effects
        """
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

        for cov in covariates:
            assert len(cov) == 2
            bool_to_size(cov[1])
            assert len(cov[0]) == np.prod(cov[1])
        self.covariates = covariates

        assert not (global_intercept and indicators != [])

        self.global_ids = global_effects_indices
        self.global_intercept = global_intercept
        self.k_beta = len(self.global_ids) + int(self.global_intercept)
        for ind in indicators:
            bool_to_size(ind)
            self.k_beta += np.prod(ind)
        self.indicators = indicators

        for ran_eff in random_effects_list:
            assert len(ran_eff) == 2
            assert all(ran_eff[1][:self.n_grouping_dims])
            bool_to_size(ran_eff[1])
        self.ran_list = random_effects_list

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

        Parameters
        ----------
        var: ndarray
            initialization for variables. If None, first run without random
            effects to obtain a starting point
        S: ndarray
            standard deviation for each measurement. The size of S should be
            the same as that of measurements vector
        uprior: ndarray
            lower and upper bounds for variables.
        trim_percentage: float
            percentage of datapoints to trim. Default is 0, i.e. no trimming.
        share_obs_std: boolean
            True if assuming data across studies share the same measurement
            standard deviation
        fit_fixed: boolean
            whether to run a fit without random effects first in order to
            obtain a good starting point
        """
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
        if self.constraints != []:
            C = lambda var: self.constraints.dot(var)

        JC = None
        if self.constraints != []:
            JC = lambda var: self.constraints

        c = None
        if self.constraints != []:
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
            print('fit with gamma fixed...')
            t0 = time.time()
            model_fixed.optimize(x0=x0,print_level=inner_print_level,max_iter=inner_max_iter)
            print('finished...elapsed',time.time()-t0)

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
        for i in range(self.n_groups):
            self.var_u.append(np.linalg.inv(np.diag(1./self.gamma_soln) +
                               np.transpose(Z_split[i]).dot(Z_split[i])/self.delta_soln))

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

        for i in range(self.n_groups):
            V = Z_split[i].dot(np.diag(self.gamma_soln)).dot(np.transpose(Z_split[i])) \
                + self.delta_soln*np.identity(self.grouping[i])
            self.var_beta += np.transpose(X_split[i]).dot(np.linalg.inv(V)).dot(X_split[i])
        self.var_beta = np.linalg.inv(self.var_beta)

    def sampleGlobalWithLimeTr(self, sample_size=100, max_iter=300):
        beta_samples,gamma_samples = LimeTr.sampleSoln(self.model, sample_size=sample_size, max_iter=max_iter)
        return beta_samples, gamma_samples

    def draw(self, n_draws=10):
        beta_samples = np.transpose(np.random.multivariate_normal(self.beta_soln, self.var_beta, n_draws))
        u_samples = [[] for _ in range(len(self.ran_list))]
        for i in range(self.n_groups):
            samples = np.random.multivariate_normal(self.u_soln[i], self.var_u[i], n_draws)
            start = 0
            for j in range(len(self.ran_list)):
                ran_eff = self.ran_list[j]
                dims = ran_eff[1]
                u_samples[j].append(samples[:,start:start + np.prod(dims[self.n_grouping_dims:])].reshape((n_draws,-1)))
                start += np.prod(dims[self.n_grouping_dims:])
        for i in range(len(u_samples)):
            u_samples[i] = np.transpose(np.hstack(u_samples[i]))
        for i in range(len(self.ran_list)):
            ran_eff = self.ran_list[i]
            _, dims = ran_eff
            u_samples[i] = u_samples[i].reshape(tuple(dims +[n_draws])).squeeze()

        return beta_samples, u_samples
