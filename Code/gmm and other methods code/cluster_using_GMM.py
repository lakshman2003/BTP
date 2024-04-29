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
# from sklearn.metrics import plot_roc_curve
import pickle

def gmm_cluster(all_data,n):
    #data and number of clusters 
    
    returns = all_data[['tradingsymbol','return']].copy()
    returns['Date'] = returns.index.copy()
    print(returns)

    transposed = returns.pivot(index = 'Date', columns = 'tradingsymbol', values = 'return')

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components = n)
    gmm.fit(transposed.dropna().transpose())
    clusters = gmm.predict(transposed.dropna().transpose())
    clusters_df = pd.DataFrame({'Cluster':clusters,
                            'Companies':transposed.columns})

    clusters_df = clusters_df.sort_values(['Cluster']).reset_index(drop = True)
    # print(clusters_df)
    all_data.to_csv(r'D:\Padantra\ulloo_client\screeners\Machine_Learning_Predictions\Intermediate_outputs\all_data.csv')
    clusters_df.to_csv(r'D:\Padantra\ulloo_client\screeners\Machine_Learning_Predictions\Intermediate_outputs\clusters_df.csv')
    print(all_data)
    
    return clusters_df
    