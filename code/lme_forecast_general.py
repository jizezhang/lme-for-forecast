import sys
path = '/Users/jizez/Dropbox (uwamath)/limetr.git/'
sys.path.insert(0, path)
from limetr import LimeTr
import numpy as np
import utils
import copy
import time


### TODO list
# - add tests
# - add examples
# - include posterior variance function

class LME:

    def __init__(self, dimensions, n_grouping_dims, measurements, covariates,
                 indicator, global_ids, ran_list,
                 global_intercept, ran_intercept):

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
        ran_intercept: list
            if not empty dimensions of random intercept excluding grouping dimensions
            e.g. if grouping dimension is location, then
            a random intercept in location would have input
            [1, 1, 1]
        """
        self.dimensions = dimensions
        self.n_grouping_dims = n_grouping_dims
        self.n_groups = np.prod(dimensions[:n_grouping_dims])
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
        self.ran_intercept = ran_intercept
        self.add_re = True
        if self.ran_list == [] and self.ran_intercept == False:
            self.add_re = False
        return


    def repeat(self,values,dims):
        """
        to compute function value
        """
        assert len(values) == np.prod(dims)
        cumprod = 1
        for i in range(self.nd-1,-1,-1):
            if dims[i] == 1:
                values = np.tile(values.reshape((-1,cumprod)),(1,self.dimensions[i])).reshape(-1)
            cumprod *= self.dimensions[i]
        assert values.shape[0] == self.N
        return values

    def repeatTranspose(self, y, dims):
        """
        to compute jacobian
        """
        values = [y]
        for i in range(self.nd):
            if dims[i] == 1:
                values = [np.sum(x.reshape((self.dimensions[i],-1)),axis=0) for x in values]
            else:
                temp = []
                for x in values:
                    temp.extend(np.split(x, self.dimensions[i]))
                values = temp
        assert len(np.squeeze(values)) == np.prod(dims)
        return np.squeeze(values)

    def kronecker(self, dims):
        """
        build Z matrix using kronecker product
        """
        # def recurse(i):
        #     if i == self.nd:
        #         return [1]
        #     if dims[i] == 1:
        #         return np.tile(recurse(i+1),(self.dimensions[i],1))
        #     else:
        #         return np.kron(np.identity(self.dimensions[i]), recurse(i+1))
        # Z = recurse(0)
        Z = [1]
        for i in range(len(dims)-1,-1,-1):
            if dims[i] == 1:
                Z = np.tile(Z,(self.dimensions[i+self.n_grouping_dims],1))
            else:
                Z = np.kron(np.identity(self.dimensions[i+self.n_grouping_dims]),Z)
        assert Z.shape[0] == np.prod(self.dimensions[self.n_grouping_dims:])
        assert Z.shape[1] == np.prod(dims)

        return Z # element-wise product

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
                y += self.repeat(values, dims)*beta[start+i]
            start += len(self.global_ids)
        if self.indicator != []:
            y += self.repeat(beta[start:start + np.prod(self.indicator)],self.indicator)
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
                val.append(np.dot(values, self.repeatTranspose(y,dims)))
        if self.indicator != []:
            val.extend(self.repeatTranspose(y,self.indicator))
        assert len(val) == self.k_beta
        return np.array(val)

    def buildZ(self):
        Z = []
        self.k_gamma = 0
        if self.ran_intercept != []:
            z = self.kronecker(self.ran_intercept)
            self.k_gamma += 1
            Z.append(np.tile(z,(self.n_groups, 1)))
        for ran in self.ran_list:
            id, dims = ran
            values = self.repeat(self.covariates[id][0],self.covariates[id][1])
            assert len(dims) + self.n_grouping_dims == self.nd
            self.k_gamma += np.prod(dims)
            Z.append(values.reshape((-1,1))*np.tile(self.kronecker(dims),(self.n_groups,1)))

        self.Z = np.hstack(Z)
        print(self.Z.shape)

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
        if self.ran_intercept:
            start += 1
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
