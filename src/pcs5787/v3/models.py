"""Referências fortes e correção binomial de logit com offset fora da amostra."""
import numpy as np
from scipy import sparse
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from ..v2.learners import Representation,make_model
import lightgbm as lgb

def safe_logit(p):return logit(np.clip(np.asarray(p),1e-8,1-1e-8))

class OffsetModel:
    def __init__(self,columns,kind='tree',calendar=True,penalty=40,leaves=7,iterations=120,seed=5787):
        self.columns=columns;self.kind=kind;self.calendar=calendar;self.penalty=penalty;self.leaves=leaves;self.iterations=iterations;self.seed=seed

    def design(self,x,fit=False):
        if fit:
            self.prep=Representation(tuple(self.columns),'strong',self.kind=='tree')
            self.prep.fit(x)
        arr=self.prep.transform(x)
        # Remove calendar variables when testing correction by weather/missingness only.
        if not self.calendar:
            cats=6 if self.kind=='tree' else sum(len(c) for c in self.prep.encoder_.categories_)
            arr=arr[:,cats:]
            if self.kind=='linear':arr=sparse.hstack([np.ones((len(x),1)),arr],format='csr')
        return arr

    def fit(self,x,y,offset):
        y=np.asarray(y,dtype=float);offset=np.asarray(offset,dtype=float);z=self.design(x,True)
        assert np.isfinite(offset).all() and set(np.unique(y)).issubset({0.,1.})
        if self.kind=='tree':
            ds=lgb.Dataset(z,label=y,init_score=offset,categorical_feature=list(range(6)) if self.calendar else 'auto',free_raw_data=False)
            params=dict(objective='binary',learning_rate=.04,num_leaves=self.leaves,min_data_in_leaf=300,lambda_l2=self.penalty,feature_fraction=.9,bagging_fraction=.85,bagging_freq=1,num_threads=4,verbosity=-1,seed=self.seed,deterministic=True,force_col_wise=True,boost_from_average=False)
            self.model=lgb.train(params,ds,num_boost_round=self.iterations)
        else:
            penalties=np.full(z.shape[1],self.penalty)
            if not self.calendar:penalties[0]=0. # Explicit calibration intercept, with regularized slope/weather.
            def loss(beta):
                eta=offset+z@beta;prob=expit(eta)
                return (np.logaddexp(0,eta)-y*eta).mean()+.5*np.sum(penalties*beta**2)/len(y),np.asarray(z.T@(prob-y)).ravel()/len(y)+penalties*beta/len(y)
            fit=minimize(loss,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10,'gtol':1e-7})
            self.beta=fit.x;self.converged=fit.success
        return self

    def correction(self,x):
        z=self.design(x)
        # LightGBM predictions contain the new trees, not Dataset.init_score.
        return self.model.predict(z,raw_score=True,num_threads=4) if self.kind=='tree' else np.asarray(z@self.beta).ravel()

    def predict(self,x,p_a,strength=1.):
        if strength==0:return np.asarray(p_a).copy()
        return expit(safe_logit(p_a)+strength*self.correction(x))
