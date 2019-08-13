import sys
path = '/Users/jizez/Dropbox (uwamath)/limetr.git/'
sys.path.insert(0, path)
from limetr import LimeTr
import numpy as np
import utils
import repeat_utils as rutils
import copy
import time


### TODO list
# - add tests
# - add examples
# - include posterior variance function

class LME:

    def __init__(self, dimensions, n_grouping_dims, measurements, covariates,
                 indicator, global_ids, ran_list,
                 global_intercept, ran_intercepts):

        """
        Parameters:
        ----------
        dimensions: list
            dimensions of the data
            the length of the list is the # of dimensions
            the value of each component specifies the size of that dimension
            e.g. if the data has dimension location-age-sex-year, the list would
            be [n_loc,n_age,n_sex,n_year]
        n_grouping_dims: int
            the number of dimensions on which grouping occurs.
            takes the first `n_grouping_dims` from `dimensions`
            e.g. if `dimensions` is [n_loc,n_age,n_sex,n_year] and
            `n_grouping_dims` is 1, then the data will be grouped by location,
            and the Z matrix corresponding to random effects will be a block
            diagonal matrix with n_loc number of blocks
        measurements: numpy array
            measurement values, must be ordered according to the `dimensions`
            e.g. if `dimensions` is [n_loc,n_age,n_sex,n_year],
            then measurements must be ordered by location, age, sex, year.
        covariates: list of tuples
            each element of the list is a tuple of (numpy.array, dimensions_of_cov)
            e.g. for HAQ, the tuple would be
            (values_of_haq, [n_loc, 1, 1, n_year])
            since HAQ does not depend on age and sex
        indicator: list
            dimensions on which to use indicator
            e.g. if age-sex indicator, the list would be
            [1, n_age, n_sex, 1]
        global_ids: list
            ids for covariates that will be used as global effects
        ran_list: list of tuples
            each element of the list is a tuple of
            (id_of_cov, dimensions_of_random_effects_excluding_grouping_dimensions)
            e.g. if we want to know random slope of covariate id=0 per location-age,
            and the grouping dimension is location, then the tuple would be
            (0,[n_age, 1, 1])
        global_intercept: boolean
            whether to use an intercept in global effects
        ran_intercepts: list of lists
            if not empty each element specifies dimensions of random intercept
            excluding grouping dimensions
            e.g. if grouping dimension is location, and random intercepts are
            location-age and location-sex, then
            [[n_age, 1, 1], [1,n_sex,1]]
        """
        self.dimensions = dimensions
        self.n_grouping_dims = n_grouping_dims
        self.n_groups = int(np.prod(dimensions[:n_grouping_dims]))
        self.grouping = [np.prod(dimensions[n_grouping_dims:])]*self.n_groups
        self.Y = measurements
        self.N = self.Y.shape[0]
        self.nd = len(dimensions)
        assert self.N == np.prod(self.dimensions)
        assert self.N == sum(self.grouping)
        self.covariates = covariates
        assert not (global_intercept and indicator != [])
        self.global_ids = global_ids
        self.indicator = indicator
        self.global_intercept = global_intercept
        self.k_beta = len(global_ids) + int(global_intercept)
        if indicator != []:
            self.k_beta += np.prod(indicator)
        self.ran_list = ran_list
        self.ran_intercepts = ran_intercepts
        self.add_re = True
        if self.ran_list == [] and self.ran_intercepts == []:
            self.add_re = False
        return

    def X(self, beta, naive=False):
        assert len(beta) == self.k_beta
        y = np.zeros(self.N)
        start = 0
        if self.global_intercept:
            y += beta[start]
            start += 1
        if len(self.global_ids) > 0:
            for i in range(len(self.global_ids)):
                ind = self.global_ids[i]
                values, dims = self.covariates[ind]
                assert values.shape[0] == np.prod(dims)
                y += rutils.repeat(values, dims, self.dimensions)*beta[start+i]
            start += len(self.global_ids)
        if self.indicator != []:
            y += rutils.repeat(beta[start:start + np.prod(self.indicator)],self.indicator, self.dimensions)
            start += np.prod(self.indicator)
        return y

    def XT(self, y):
        assert len(y) == self.N
        start = 0
        val = []
        if self.global_intercept:
            val.append(np.sum(y))
        if len(self.global_ids) > 0:
            for i in self.global_ids:
                values, dims = self.covariates[i]
                val.append(np.dot(values, rutils.repeatTranspose(y,dims, self.dimensions)))
        if self.indicator != []:
            val.extend(rutils.repeatTranspose(y,self.indicator, self.dimensions))
        assert len(val) == self.k_beta
        return np.array(val)

    def buildZ(self):
        Z = []
        self.k_gamma = 0
        if self.ran_intercepts != []:
            for intercept in self.ran_intercepts:
                z = rutils.kronecker(intercept, self.dimensions, self.n_grouping_dims)
                Z.append(np.tile(z,(self.n_groups, 1)))
            self.k_gamma += len(self.ran_intercepts)
        for ran in self.ran_list:
            id, dims = ran
            values = rutils.repeat(self.covariates[id][0],self.covariates[id][1], self.dimensions)
            assert len(dims) + self.n_grouping_dims == self.nd
            self.k_gamma += np.prod(dims)
            Z.append(values.reshape((-1,1))*np.tile(rutils.kronecker(dims, self.dimensions, self.n_grouping_dims),(self.n_groups,1)))

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
        if self.ran_intercepts != []:
            start += len(self.ran_intercepts)
        for ran in self.ran_list:
            _, dims = ran
            if np.prod(dims) > 1:
                c = np.zeros((np.prod(dims)-1,k))
                for i in range(np.prod(dims)-1):
                    c[i,start+i] = 1
                    c[i,start+i+1] = -1
                C.append(c)
            start += np.prod(dims)
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
                self.info = model_fixed.info
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
        assert len(self.ran_list) > 0 or len(self.ran_intercepts) > 0
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

        if len(self.global_ids) > 0:
            for i in range(len(self.global_ids)):
                ind = self.global_ids[i]
                values, dims = self.covariates[ind]
                assert values.shape[0] == np.prod(dims)
                X[:,start] = rutils.repeat(values, dims, self.dimensions)
                start += 1

        if self.indicator != []:
            X[:,start:] = rutils.kronecker(self.indicator, self.dimensions, 0)

        X_split = np.split(X, self.n_groups)

        for i in range(self.n_groups):
            V = Z_split[i].dot(np.diag(self.gamma_soln)).dot(np.transpose(Z_split[i])) \
                + self.delta_soln*np.identity(self.grouping[i])
            self.var_beta += np.transpose(X_split[i]).dot(np.linalg.inv(V)).dot(X_split[i])
        self.var_beta = np.linalg.inv(self.var_beta)

    def sampleGlobalWithLimeTr(self, sample_size=100, max_iter=300):
        beta_samples,gamma_samples = LimeTr.sampleSoln(self.model, sample_size=sample_size, max_iter=max_iter)
        return beta_samples, gamma_samples
