from . import visualize as vs
from . import stats
import numpy as np

#volatility clustering
def acf(x,file_name,for_abs=True,multiple=False,fit=False,scale='log',max_lag=1000):
    if for_abs:
        x = np.abs(x)
    if multiple:
        res = np.zeros(max_lag)
        for e in x:
            res += stats.acf(e, max_lag=max_lag)
        res /= x.shape[0]
    else:
        res = stats.acf(x, max_lag=max_lag)
    vs.acf(res,file_name,scale=scale)
    return res

def leverage_effect(x,file_name,multiple=True,min_lag=1,max_lag=100):
    def compute_levs(x,x_abs):
        Z = (np.mean(x_abs**2))**2
        second_term = np.mean(x)*np.mean(x_abs**2)
        def compute_for_t(t):
            if t == 0:
                first_term = np.mean(x*(x_abs)**2)
            elif t > 0:
                first_term = np.mean(x[:-t]*(x_abs[t:]**2))
            else:
                first_term = np.mean(x[-t:]*(x_abs[:t]**2) )
            return (first_term-second_term)/Z
        levs = [compute_for_t(t) for t in range(min_lag,max_lag)]
        return np.array(levs)

    x_abs = np.abs(x) # 1. compute absolute returns
    if multiple:
        levs = np.zeros(max_lag-min_lag) #2. aggregate across all stocks if multiple = True
        for e1,e2 in zip(x,x_abs): # for each stock iterate
            levs += compute_levs(e1,e2)
        levs /= x.shape[0] # taking the mean
    else:
        levs = compute_levs(x,x_abs)
    vs.leverage_effect([i for i in range(min_lag,max_lag)],levs,file_name)
    return levs


#fat-tail
def distribution(x,file_name,scale='linear',multiple=False,normalize=True,granuality=100):
    #preprocessing
    if multiple:
        x = np.reshape(x,x.size)
    if normalize:
        x = normalize_time_series(x)
    if scale == 'linear':
        dist_x,dist_y = linear_pdf(x,granuality=granuality)
        vs.distribution(dist_x, dist_y, file_name, 'linear')
        return dist_x, dist_y
    elif scale == 'log':
        dist_x,dist_y = linear_pdf(x,granuality=granuality)
        vs.distribution(dist_x, dist_y, file_name, 'log')
        return dist_x, dist_y
    else:
        pass

def culmulative_distribution(x,scale='linear',normalize=True):
    pass

def normalize_time_series(x):
    mean = np.mean(x)
    std = np.std(x)
    x = (x-mean)/std
    return x

def linear_pdf(x,dist_x=None,granuality=100):
    if dist_x is None:
        x_max = 6.
        x_min = -6.
    dist_x = np.linspace(x_min,x_max,granuality)
    diff = dist_x[1]-dist_x[0]
    dist_x_visual = (dist_x + diff)[:-1]
    dist_y = np.zeros(granuality-1)
    for e,(x1,x2) in enumerate(zip(dist_x[:-1],dist_x[1:])):
        dist_y[e] = x[np.logical_and(x > x1,x < x2)].size
    dist_y /= x.size
    return dist_x_visual,dist_y


def log_pdf(x,dist_x=None,granuality=100):
    pass

def cdf(x,scale='linear'):
    pass

