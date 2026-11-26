"""Transformações treináveis e famílias de modelos; lista explícita de entradas."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier,LGBMRegressor


class Representation(BaseEstimator,TransformerMixin):
    def __init__(self,numeric=(),calendar='strong',tree=False,anomaly=False,interactions=False,splines=False):
        self.numeric=numeric;self.calendar=calendar;self.tree=tree;self.anomaly=anomaly;self.interactions=interactions;self.splines=splines

    def _raw(self,x,fit=False):
        z=x[list(self.numeric)].to_numpy(dtype=float) if self.numeric else np.empty((len(x),0))
        if self.anomaly:
            out=[]
            if fit:self.climate_={}
            for c in ['wind_lag1','gust_lag1','sp_wind','sp_gust']:
                if c not in self.numeric:continue
                key=x.region.astype(str)+'_'+x.month.astype(str)+'_'+x.hour.astype(str)
                if fit:
                    d=pd.DataFrame({'key':key.to_numpy(),'region':x.region.to_numpy(),'value':x[c].to_numpy()})
                    region=d.groupby('region').value.mean();global_mean=d.value.mean()
                    stats=d.groupby('key').value.agg(['mean','count'])
                    prior=stats.index.to_series().str.split('_').str[0].map(region).fillna(global_mean)
                    means=(stats['mean']*stats['count']+30*prior)/(stats['count']+30)
                    self.climate_[c]=(means,region,global_mean)
                means,region,global_mean=self.climate_[c]
                expected=key.map(means).fillna(x.region.map(region)).fillna(global_mean).to_numpy()
                out.append(x[c].to_numpy()-expected)
            if out:z=np.column_stack([z,*out])
        return z

    def fit(self,x,y=None):
        self.cat_=['region','hour','weekday','month']+(['region_hour','region_month'] if self.calendar=='strong' else [])
        self.encoder_=(OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan) if self.tree else OneHotEncoder(handle_unknown='ignore',sparse_output=True))
        self.encoder_.fit(x[self.cat_].astype(str))
        z=self._raw(x,True)
        self.num_count_=z.shape[1]
        if self.num_count_:
            with __import__('warnings').catch_warnings():
                __import__('warnings').simplefilter('ignore',RuntimeWarning);self.medians_=np.nanmedian(z,axis=0)
            self.medians_=np.nan_to_num(self.medians_,nan=0)
            clean=np.where(np.isfinite(z),z,self.medians_)
            self.scaler_=StandardScaler().fit(clean)
            if self.splines:
                self.spline_indices_=[i for i,c in enumerate(self.numeric) if c in ['wind_lag1','gust_lag1','wind_mean24','sp_wind','sp_gust','sp_wind_std'] and np.std(clean[:,i])>1e-8]
                if self.spline_indices_:self.spline_=SplineTransformer(n_knots=5,degree=3,knots='quantile',extrapolation='linear',include_bias=False).fit(clean[:,self.spline_indices_])
            if self.interactions:self.region_encoder_=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(x[['region']])
        return self

    def transform(self,x):
        cat=self.encoder_.transform(x[self.cat_].astype(str))
        if not self.num_count_:return cat
        z=self._raw(x);missing=~np.isfinite(z);clean=np.where(~missing,z,self.medians_)
        if self.tree:
            return np.column_stack([cat,z,missing.astype(float)]) # árvores tratam ausência nativamente
        scaled=self.scaler_.transform(clean);parts=[cat,sparse.csr_matrix(scaled),sparse.csr_matrix(missing)]
        if self.splines and self.spline_indices_:parts.append(sparse.csr_matrix(self.spline_.transform(clean[:,self.spline_indices_])))
        if self.interactions:
            reg=self.region_encoder_.transform(x[['region']])
            k=min(len(self.numeric),12)
            parts.append(sparse.csr_matrix((reg[:,:,None]*scaled[:,None,:k]).reshape(len(x),-1)))
        return sparse.hstack(parts,format='csr')


def make_model(spec,columns,regression=False):
    algorithm=spec['algorithm'];params=spec.get('params',{});seed=spec.get('seed',5787)
    tree=algorithm in ['histgb','lightgbm']
    prep=Representation(tuple(columns),spec.get('calendar','strong'),tree,spec.get('anomaly',False),algorithm=='logistic_interaction',algorithm=='logistic_spline')
    cats=list(range(6 if spec.get('calendar','strong')=='strong' else 4))
    if algorithm.startswith('logistic'):
        model=LogisticRegression(C=params.get('C',1),solver='lbfgs',max_iter=params.get('max_iter',600),tol=1e-5,random_state=seed)
    elif algorithm=='histgb':
        model=HistGradientBoostingClassifier(max_iter=params.get('iterations',200),learning_rate=params.get('learning_rate',.06),max_leaf_nodes=params.get('leaves',15),min_samples_leaf=params.get('min_leaf',60),l2_regularization=params.get('l2',5),categorical_features=cats,early_stopping=False,random_state=seed)
    elif algorithm=='lightgbm':
        cls=LGBMRegressor if regression else LGBMClassifier
        model=cls(objective=params.get('objective','regression' if regression else 'binary'),n_estimators=params.get('iterations',250),learning_rate=params.get('learning_rate',.05),num_leaves=params.get('leaves',15),min_child_samples=params.get('min_leaf',100),reg_lambda=params.get('l2',5),colsample_bytree=params.get('colsample',.9),subsample=.85,subsample_freq=1,n_jobs=4,verbosity=-1,random_state=seed,deterministic=True,force_col_wise=True)
    else:raise ValueError(algorithm)
    return Pipeline([('representation',prep),('model',model)])
