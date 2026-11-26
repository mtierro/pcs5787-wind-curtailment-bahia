"""Calibração monotônica usando somente probabilidades OOF anteriores disponíveis."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize,brentq
from sklearn.metrics import f1_score
from .common import cutoff


def fit_calibrator(y,p,method):
    y=np.asarray(y,dtype=float);z=logit(np.clip(p,1e-8,1-1e-8))
    if method=='none' or len(y)<100 or len(np.unique(y))<2:return {'method':method,'slope':1.,'intercept':0.,'n':len(y),'fallback':method!='none'}
    if method=='intercept':
        intercept=brentq(lambda a:expit(z+a).mean()-y.mean(),-40,40);slope=1.
    elif method=='monotone_platt':
        def loss(theta):
            q=theta[0]*z+theta[1]
            return np.mean(np.logaddexp(0,q)-y*q)+1e-5*(theta[0]-1)**2
        result=minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(.05,10),(-20,20)])
        assert result.success,result.message
        slope,intercept=result.x
    else:raise ValueError(method)
    return {'method':method,'slope':float(slope),'intercept':float(intercept),'n':len(y),'fallback':False}


def apply_calibrator(p,fit):
    if fit['slope']==1 and fit['intercept']==0:return np.asarray(p)
    return expit(fit['slope']*logit(np.clip(p,1e-8,1-1e-8))+fit['intercept'])


def available_history(predictions,origin,days=90,latency=None):
    end=cutoff(origin,latency)
    return predictions[predictions.time_local.lt(end)&predictions.time_local.ge(end-pd.Timedelta(days=days))]


def decision_threshold(y,p):
    if len(y)<100 or np.sum(y)==0:return .5
    grid=np.arange(.01,1,.01)
    return float(grid[np.argmax([f1_score(y,np.asarray(p)>=t,zero_division=0) for t in grid])])
