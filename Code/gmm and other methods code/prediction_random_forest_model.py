# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:11:12 2022

@author: PRIYAM 
"""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as pdr
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from random import sample
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.cluster import SpectralClustering
import pandas_datareader as pdr
import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import sys
from IPython.display import clear_output
import os
from sklearn import cluster, preprocessing
from scipy.optimize import fsolve, minimize, basinhopping
# import cvxopt as opt
# from cvxopt import blas, solvers
# import ipywidgets as widgets
import time
import bs4 as bs
import pickle
import requests
from sklearn.linear_model import LinearRegression
# from guppy import hpy
# import quandl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, validation_curve, TimeSeriesSplit
from sklearn.metrics import plot_roc_curve
import pickle

def predictions(clusters_df,all_data,skip_days = 7, Start_Trade_Date = '2019-01-02'):
    print(all_data)
    pred_for_tomorrow = pd.DataFrame({'Date':[],
                                    'company':[],
                                    'prediction':[]})
    # clusters_df

    
    data = all_data[Start_Trade_Date:]
    dates = data.index.unique()
    print("data",data)
    seven_days_skipped = dates[::skip_days]
    print("huehuehue")
    print(seven_days_skipped)

    for i in range(len(seven_days_skipped)-1):
        print(i)
        
        ##Get Prediction for Tomorrow##
        date = seven_days_skipped[i]
        print(date)

        for cluster_selected in clusters_df.Cluster.unique():
            rf_cv =  pickle.load(open(os.getcwd() + f'\\model\\Cluster_{cluster_selected}', 'rb'))
            best_rf = rf_cv.best_estimator_
            cluster_data = data.loc[data.tic.isin(clusters_df.loc[clusters_df.Cluster==cluster_selected,'Companies'].tolist())].loc[[date]].copy()
            cluster_data = cluster_data.dropna()
            if (cluster_data.shape[0]>0):
                X_test = cluster_data[['atr', 'bbw','obv','cmf','macd', 'adx', 'sma', 'ema', 'cci', 'rsi']]
                print("X_test:",X_test)
                print(rf_cv.predict_proba(X_test))
               # print(rf_cv.predict_log_proba(X_test))
                pred_for_tomorrow = pred_for_tomorrow.append(pd.DataFrame({'company':cluster_data['tic'],
                                                                            'prediction':rf_cv.predict_proba(X_test)[:,1],
                                                                            'Date':cluster_data.index}), ignore_index = True)
            else:
                continue
        pred_for_tomorrow = pred_for_tomorrow.sort_values(by = ['prediction'], ascending= False).reset_index(drop = True)

        i += 1
        pred_for_tomorrow.to_csv("predictions.csv")
        print(pred_for_tomorrow)
    return pred_for_tomorrow