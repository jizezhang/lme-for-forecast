import sys
path = '/Users/jizez/Dropbox (uwamath)/limetr.git/'
sys.path.insert(0, path)
from limetr import LimeTr
import numpy as np
import utils
import copy
import time


class LME:

    def __init__(self, measurements, covariates, fix_cov_ids=[], ran_cov_ids=[],
                 n_year=28, n_loc=195, n_age=23, n_sex=2,
                 fix_intercept=False, ran_intercept=True,
                 age_sex=True, location_age=False):
        """
        Create a linear mixed effects (LME) model for forecast.

        Parameters
        ----------
        measurements: ndarray
            1-D ndarray that stores values of measurements
        covariates: ndarray
            2-D ndarray that stores values of covariates
        fix_cov_ids: list
            columns of covariates that will be used as fixed effects covs
        ran_cov_ids: list
            columns of covariates that will be used as random effects covs
        n_year: int
            number of years
        n_loc: int
            number of locations
        n_age: int
            number of age groups
        n_sex: int
            number of sexes
        fix_intercept: boolean
            whether to include an intercept in fixed effects covs
        ran_intercept: boolean
            whether to include an intercept in random effects covs
        age_sex: boolean
            whether to include age_sex indicators
        location_age: boolean
            whether grouping is by location-age. Default is by location only.

        """
        assert not(fix_intercept and age_sex)
        self.T = n_year
        self.n_loc = n_loc
        self.n_age = n_age
        self.n_sex = n_sex
        self.age_sex = age_sex
        self.location_age = location_age
        self.fix_intercept = fix_intercept

        self.Y = measurements
        assert self.Y.shape[0] == covariates.shape[0]
        self.N = self.Y.shape[0]
        self.X = covariates[:,fix_cov_ids] # no intercept or age_sex_indicator
        self.k_beta = len(fix_cov_ids)
        if self.fix_intercept:
            self.k_beta += 1
        if self.age_sex:
            self.k_beta += self.n_age*self.n_sex

        if len(ran_cov_ids) == 0 and ran_intercept == False:
            self.add_re = False
            self.Z = np.ones(self.N).reshape((-1,1))
            print('no random effects')
        else:
            self.add_re = True
            self.k_gamma = len(ran_cov_ids)
            if ran_intercept == True:
                self.k_gamma += 1
                self.Z = np.ones(self.N).reshape((-1,1))
                if len(ran_cov_ids) > 0:
                    #self.Z = np.hstack((covariates[:,ran_cov_ids].reshape((-1,len(ran_cov_ids))), self.Z))
                    self.Z = np.insert(covariates[:,ran_cov_ids].reshape((-1,len(ran_cov_ids))),0,1,axis=1)
            else:
                self.Z = covariates[:,ran_cov_ids].reshape((-1,len(ran_cov_ids)))

        self.grouping = [self.T*self.n_age*self.n_sex]*self.n_loc
        if self.location_age:
            self.grouping = [self.T*self.n_sex]*(self.n_age*self.n_loc)
        self.n_group = len(self.grouping)

    def check(self):
        print('number of datapoints', self.N)
        print('number of studies', self.n_group)
        print('number of global effects coefs', self.k_beta)
        print('number of random effects coefs', self.k_gamma)

    def age_sex_func(self,beta):
        """
        Each intercept value is repeated T times where T corresponds to the length
        of a time series. Then the entire vector is repeated n_loc times where n_loc
        corresponds to the number of locations.
        """
        return np.tile(np.repeat(beta,self.T),self.n_loc)

    def age_sex_jacob(self,y):
        """
        1. divide and sum up every subvector of length T (corresponds to the length
           of a time series) in y
        2. divide and sum up same age-sex value across all locations
        - The default ordering of reshape((m,n)) creates a matrix with m rows and
          n cols.
        """
        return np.sum(np.sum(y.reshape((-1,self.T)),axis=1).reshape((-1,self.n_age*self.n_sex)),axis=0).squeeze()

    def optimize(self, var=None, S=None, uprior=None, trim_percentage=0.0,
                 share_obs_std=True, fit_fixed=True,inner_print_level=5,
                 inner_max_iter=50, outer_verbose=False,
                 outer_max_iter=100, outer_step_size=1,
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
        self.check()
        k = self.k_beta + self.k_gamma
        if S is None:
            if share_obs_std:
                k += 1
            else:
                k += len(grouping)
        print('total number of fixed effects variables',k)


        up = []
        if uprior is None:
            up = np.array([
                [-np.inf]*self.k_beta + [1e-8]*self.k_gamma +\
                    [1e-7]*(k-self.k_beta-self.k_gamma),
                [np.inf]*k
                ])
        else:
            up = uprior

        # def fX(beta):
        #     val = np.zeros(self.N)
        #     start = 0
        #     if self.age_sex:
        #         val += self.age_sex_func(beta[:self.n_age*self.n_sex])
        #         start += self.n_age*self.n_sex
        #     if self.k_beta > 0:
        #         val += np.dot(self.X,beta[start:start+self.k_beta])
        #     if self.fix_intercept:
        #         val += beta[-1]
        #     return val
        #
        # def jX(y):
        #     if self.age_sex:
        #         return np.append(self.age_sex_jacob(y),np.transpose(self.X).dot(y))
        #     else:
        #         return np.append(np.transpose(self.X).dot(y),[np.sum(y)])

        def fX(beta):
            val = np.zeros(self.N)
            start = 0
            if self.fix_intercept:
                val += beta[0]
                start += 1
            if self.age_sex:
                val += self.age_sex_func(beta[:self.n_age*self.n_sex])
                start += self.n_age*self.n_sex
            if self.k_beta > 0:
                val += np.dot(self.X,beta[start:start+self.k_beta])
            return val

        def jX(y):
            if self.age_sex:
                return np.append(self.age_sex_jacob(y),np.transpose(self.X).dot(y))
            elif self.fix_intercept:
                return np.insert(np.transpose(self.X).dot(y),0,np.sum(y))
            else:
                return np.transpose(self.X).dot(y)

        x0 = np.ones(k)*.1
        if var is not None:
            assert len(var) == k
            x0 = var

        if var is None or fit_fixed or self.add_re == False:
            uprior_fixed = copy.deepcopy(up)
            uprior_fixed[:,self.k_beta:self.k_beta+self.k_gamma] = 0.0
            model_fixed = LimeTr(self.grouping, self.k_beta, self.k_gamma, self.Y, fX, jX, \
                                 self.Z, S=S,inlier_percentage=1.-trim_percentage,\
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

        model = LimeTr(self.grouping, self.k_beta, self.k_gamma, self.Y, fX, jX, \
                       self.Z, S=S, inlier_percentage=1-trim_percentage,\
                       share_obs_std=share_obs_std, uprior=up)

        model.fitModel(x0=x0,
                     inner_print_level=inner_print_level,
                     inner_max_iter=inner_max_iter,
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
        Z_split = np.split(self.Z,self.n_group)
        yfit_no_random_split = np.split(self.yfit_no_random, self.n_group)

        for i in range(self.n_group):
            self.yfit.append(yfit_no_random_split[i] + Z_split[i].dot(self.u_soln[i]))
        self.yfit = np.concatenate(self.yfit)
        self.model = model

    def postVarRandom(self):
        """
        y_k = X_k beta + Z_k u_k + epsilon, u has var_mat D, epsilon has var_mat R

        Var(u_k) = inv(inv(D) + Z_k'inv(R)Z_k)
        """
        Z_split = np.split(self.Z,self.n_group)
        self.var_u = []
        for i in range(self.n_group):
            self.var_u.append(np.linalg.inv(np.diag(1./self.gamma_soln) +
                               np.transpose(Z_split[i]).dot(Z_split[i])/self.delta_soln))

    def postVarGlobal(self):
        """
        y_k = X_k beta + Z_k u_k + epsilon, u has var_mat D, epsilon has var_mat R

        Var(beta) = inv( sum_k X_k' inv(Z_k*D*Z_k' + R) X_k )

        """
        assert self.k_beta > 0
        X_split = np.split(self.X,self.n_group)
        Z_split = np.split(self.Z,self.n_group)

        self.var_beta = np.zeros((self.k_beta, self.k_beta))

        if self.fix_intercept == True:
            X_split = np.split(np.insert(self.X,0,1,axis=1), self.n_group)

        if self.age_sex == True:
            age_sex_mat = np.kron(np.identity(self.n_age*self.n_sex), np.ones((self.T,1)))
            if self.location_age == False:
                for i in range(self.n_group):
                    X_split[i] = np.hstack((age_sex_mat,X_split[i]))
            else:
                for i in range(self.n_group):
                    m = i%self.n_age
                    X_split[i] = np.hstack((age_sex_mat[m*(self.T*self.n_sex):(m+1)*(self.T*self.n_sex),:],X_split[i]))

        for i in range(self.n_group):
            V = Z_split[i].dot(np.diag(self.gamma_soln)).dot(np.transpose(Z_split[i])) \
                + self.delta_soln*np.identity(self.grouping[i])
            self.var_beta += np.transpose(X_split[i]).dot(np.linalg.inv(V)).dot(X_split[i])
        self.var_beta = np.linalg.inv(self.var_beta)

    def sampleGlobalWithLimeTr(self, sample_size=100):
        beta_samples,_ = LimeTr.sampleSoln(self.model, sample_size=sample_size)
        return beta_samples, np.mean(beta_samples,axis=0)
