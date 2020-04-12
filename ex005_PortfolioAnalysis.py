# standard imports
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
%matplotlib inline

# normal distribution
from scipy.stats import norm

# optimization and regression imports
from scipy.optimize import minimize
from scipy import stats

# data imports
import pandas_datareader as web
# IEX data import
import os
os.environ["IEX_API_KEY"] = ""
# Quandl data import
import quandl
quandl.ApiConfig.api_key = ""


###############################################################################################
# yahoo data 5 years rolling if no start/end
# spy_etf = web.DataReader('SPY','yahoo')
start = datetime.date(2019,1,1)
end = datetime.date.today()

# list of stocks in portfolio
# ALPHABETICAL ORDER Required
# '0001.HK', '1038.HK', 
stocks = ['AAPL', 'ACN', 'AMZN', 
          'COST', 'DIS', 'GOOG', 'GS', 'FB', 
          'IRBT', 'MAR', 'MMM', 'MSFT', 'NFLX', 
          'SBNY', 'SBUX', 'SHAK', 'SPY', 'UBS', 
          'VGSIX', 'VIRT']

"""
spy_etf = web.DataReader('SPY','yahoo',start,end)
amzn = web.DataReader('AMZN','yahoo',start,end)
sbny = web.DataReader('SBNY','yahoo',start,end)
"""

# download daily price data for each of the stocks in the portfolio
data = web.DataReader(stocks,'yahoo',start,end)['Adj Close']
data_volume = web.DataReader(stocks,'yahoo',start,end)['Volume']

data.sort_index(inplace=True)

# save down the data
data.to_csv('data.csv',sep=',')

# convert daily stock prices into daily returns
# data might have calendar gaps but the change automatically fills it in with 0
returns = data.pct_change(1)
returns.to_csv('returns.csv',sep=',')

# log returns
log_ret = np.log(data/data.shift(1))

# calculate mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

# set array holding portfolio weights of each stock
#weights = np.asarray([0.5,0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
weights = np.ones(20)*0.05

# returns.plot(figsize=(12,8))
#returns['VIRT'].plot(kind='hist', bins=50,figsize=(8,7),color='green')
#returns['AAPL'].plot(kind='hist', bins=50,figsize=(8,7),color='red')

# stacked histogram
# returns.plot(kind='hist',bins=30,figsize=(10,7))
# separate histograms
log_ret.hist(bins=30,figsize=(12,8), color='c', edgecolor='k', alpha=0.65)
plt.tight_layout()


###############################################################################################
# covariance matrix
# daily
log_ret.cov()

# daily to annually
(log_ret.cov() * 252).round(decimals=2)



###############################################################################################
# Single Stock Distribution with Mean - Method 2
# Much easier normal distribution using Seaborns

# Variable
stock_log_ret = log_ret['MSFT']

plt.figure(figsize=(12,5))
sns.distplot(stock_log_ret.dropna(),bins=20,fit=norm) #bins=None is another option

# Median Line
plt.axvline(stock_log_ret.median(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(stock_log_ret.median()*1.1, max_ylim*0.9, 'Median = {:.4f}'.format(stock_log_ret.median()))
min_xlim, max_xlim = plt.xlim()




###############################################################################################
# Box Plots
# color = {'boxes': 'DarkGreen', 'whiskers': 'Blue', 'medians': 'DarkBlue', 'caps': 'Gray'}
# returns.plot(kind='box',vert=False, color=color, figsize=(10,7))
plt.style.use('seaborn-whitegrid') # ggplot  tableau-colorblind10  seaborn-dark-palette  bmh  default

"""
# Method 1 - works - no custom colored plots
bplot1 = returns.plot(kind='box',vert=False, notch=True, figsize=(10,7))  # works

# Method 2 - works - pandas.DataFrame.boxplot - filled plots - no custom color
bplot1 = returns.boxplot(vert=False, notch=True, figsize=(10,7), patch_artist=True) # works
bplot1.set_title('Title')
bplot1.set_xlabel('Observed Returns')
bplot1.set_ylabel('Companies')

# Method 3 - works - no mean() dot
fig, ax = plt.subplots(figsize=(10,6))
bplot1 = ax.boxplot(returns.dropna().values, patch_artist=True, vert=False, labels=returns.columns)
bplot1 = ax.boxplot(returns.dropna().values, patch_artist=True, vert=False, labels=returns.columns, notch=True)
bplot1 = ax.boxplot(returns.dropna().values, patch_artist=False, vert=False, labels=returns.columns)
                    
# Method 4 - works - has mean
fig, ax = plt.subplots(figsize=(10,6))
# mean dots could be o or d
meanpointprops = dict(marker='o', markeredgecolor='white', markersize=3, markerfacecolor='black')
# box plot
bplot1 = ax.boxplot(returns.dropna().values, patch_artist=True, vert=False, 
                    labels=returns.columns, 
                    meanprops=meanpointprops, meanline=False, showmeans=True, whis=[5,95]) # 
# boxplot fill with colors
colors = ['pink', 'lightblue', 'lightgreen', 'darkblue','darkgoldenrod','firebrick', 'DarkBlue']
colors = []
for s in list(returns.columns):
    if returns[s].median() > 0:
        colors.append('lightblue')
    else:
        colors.append('firebrick')
# set the colors
for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
plt.setp(bplot1['medians'], color='orange')

        
"""

fig, ax = plt.subplots(figsize=(10,6))

# mean dots could be o or d
meanpointprops = dict(marker='o', markeredgecolor='white', markersize=3, markerfacecolor='black')
# box plot
bplot1 = ax.boxplot(returns.dropna().values, patch_artist=True, vert=False, 
                    labels=returns.columns, 
                    meanprops=meanpointprops, meanline=False, showmeans=True, whis=[5,95]) # [5,95] or 1.5 default
# whis = determines the reach of the whiskers to the beyond the first and third quartiles. 
# In other words, IQR is the interquartile range (Q3-Q1), the upper whisker will extend to last datum less than Q3 + whis*IQR). 
# Similarly, the lower whisker will extend to the first datum greater than Q1 - whis*IQR. 
# Beyond the whiskers, data are considered outliers and are plotted as individual points. 
# Alternatively, set this to an ascending sequence of percentile (ex.[5, 95]) to set whiskers at specific percentiles of the data 

# boxplot fill with colors
colors = ['pink', 'lightblue', 'lightgreen', 'darkblue','darkgoldenrod','firebrick', 'DarkBlue']
colors = []
for s in list(returns.columns):
    if returns[s].median() > 0:
        colors.append('lightblue')
    else:
        colors.append('firebrick')
# set the colors
for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
plt.setp(bplot1['medians'], color='orange')


# clean up script
ax.set_title('Box Plot - Percentage Returns', fontsize=10)
ax.set_xlabel('Observed Return')
ax.set_ylabel('Companies')

plt.tight_layout()
#fig.suptitle("Super Title")
#fig.subplots_adjust(hspace=0.1)
#plt.savefig('boxplot_whis_5_95.png')
plt.show()


###############################################################################################
# Basic Sharpe Ratio Calculation

# to get same random number each time
np.random.seed(101)

# remind us which stocks are there
print(log_ret.columns)

# Weights - Generate 4 random weights
weights = np.array(np.random.random(len(log_ret.columns)))
print("Random Weights")
print(weights)
# Weights - Normalization technique to sum to 1
print('Rebalance')
weights = weights/np.sum(weights)
print(weights)

# Expected Portfolio Return, adjust for risk free rate
print('Risk Free Rate')
rf_rate = 0.0   # Annualized risk free rate
print(rf_rate)

# Expected Portfolio Return = take log returns, then average it, then, x weights x 252 trading days, then sum it up
print('Expected Portfolio Return')
exp_ret = np.sum( (log_ret.mean() * weights) * 252)  - rf_rate
print(exp_ret)


# Expected volatility (denominator of sharpe ratio) = Calculate Variance/StdDev
# First take covariance of the log returns, annualize by multiply by 252
# Then dot product the covariance with the weights
# Then dot product again with the weights transpose
# Then square root the whole thing
# Use numpy dot products to run faster (can re-write to run slower so use linear algebra speed complex code)
print('Expected Volatility')
exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
print(exp_vol)

# Sharpe Ratio
# The greater the Sharpe ratio, the more attractive the risk-adjusted return is, used to compare investment strategies
# Sharpe ratio is useful to determine how much risk is being taken to achieve a certain level of return
# 1) Assumption that returns are normally distributed, but in real market scenarios, 
# the distribution might suffer from kurtosis and fatter tails, which decreases the relevance of its use
# 2) Drawback of Sharpe ratio is that it cannot distinguish between upside and downside and focuses on volatility 
# but not its direction
# 3) Sharpe ratio is backward-looking and accounts for historical returns & volatility, can be manipulated on lookback window
# 4) In future, look at Sortino ratio for downside only
print('Sharpe Ratio')
SR = exp_ret/exp_vol
print(SR)


###############################################################################################
# For Loop Sharpe Ratio (No Print Statements)
# Basic Sharpe Ratio Calculation

# to get same random number each time
# np.random.seed(101)

# Number of portfolios
num_ports = 5000

# Save results through each iteration
# Create a 2D array of weights (rows=num_ports, columns=number_stocks)
all_weights = np.zeros((num_ports,len(log_ret.columns))) 
# array to hold all the returns, vols, SRs
ret_arr = np.zeros(num_ports)                            
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

# For loop
for ind in range(num_ports):

    # Weights
    weights = np.array(np.random.random(  len(log_ret.columns)  ))
    # Normalization technique to sum to 1
    weights = weights/np.sum(weights)
    # For Loop - Save weights 
    all_weights[ind,:] = weights

    # Expected Portfolio Return, adjust for risk free rate
    rf_rate = 0.0   # Annualized risk free rate

    # Expected Portfolio Return = take log returns, then average it, then, x weights x 252 trading days, then sum it up
    # exp_ret = np.sum( (log_ret.mean() * weights) * 252)  - rf_rate
    # for loop saving:
    ret_arr[ind] = np.sum( (log_ret.mean() * weights) * 252)  - rf_rate

    # Expected volatility (denominator of sharpe ratio) = Calculate Variance/StdDev
    # First take covariance of the log returns, annualize by multiply by 252
    # Then dot product the covariance with the weights
    # Then dot product again with the weights transpose
    # Then square root the whole thing
    # Use numpy dot products to run faster (can re-write to run slower so use linear algebra speed complex code)
    # exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))

    # Sharpe Ratio
    # The greater the Sharpe ratio, the more attractive the risk-adjusted return is, used to compare investment strategies
    # Sharpe ratio is useful to determine how much risk is being taken to achieve a certain level of return
    # 1) Assumption that returns are normally distributed, but in real market scenarios, 
    # the distribution might suffer from kurtosis and fatter tails, which decreases the relevance of its use
    # 2) Drawback of Sharpe ratio is that it cannot distinguish between upside and downside and focuses on volatility 
    # but not its direction
    # 3) Sharpe ratio is backward-looking and accounts for historical returns & volatility, can be manipulated on lookback window
    # 4) In future, look at Sortino ratio for downside only
    # SR = exp_ret/exp_vol
    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]
    
# Highest sharpe ratio is 
print('Highest Sharpe Ratio:')
print(sharpe_arr.max())

# Location of highest sharpe ratio
print('Row Location of Highest Sharpe Ratio:')
print(sharpe_arr.argmax()) # for example portfolio it is row 1420

# So the weight would be
max_sr_location = sharpe_arr.argmax()  # 1420

# all_weights[1420,:]
print('Weights of Max Row:')
print(all_weights[max_sr_location,:])
all_weights[max_sr_location,:]
max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

# Scatter plot out the 
fig, ax = plt.subplots(figsize=(12,8))
scatter = ax.scatter(vol_arr, ret_arr,c=sharpe_arr,cmap='plasma')  # cmap='plasma'
# color bar for a fig subplot 
fig.colorbar(scatter, label="Sharpe Ratio")


# Scatter plot the max point, red dot
ax.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')

ax.set_xlabel('Volatility')
ax.set_ylabel('Return')

###############################################################################################
# Optimization Function
# Previously random
# Use scipy to help

# log returns
log_ret = np.log(data/data.shift(1))

# function that returns [Return, Volatility, Sharpe Ratio]
# input is weight matrix
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252,weights))) # do all calcs before sqrt
    sr = ret/vol
    return np.array([ret,vol,sr])

# Constraints for Minimizer
# Constraints mean less things to check

# This function is a helper function for scipy minimizer (to calculate the negative sharpe ratio)
def neg_sharpe(weights):
    # get_ret_vol_sr(weights) = np.array([pret,vol,sr])
    return get_ret_vol_sr(weights)[2]*-1


# This function is a constraint function to check the sum of the weights = 1
# return 0 if sum of the weights is 1
def check_sum(weights):
    return np.sum(weights) - 1

# Other Constraints
constraint = ({'type':'eq','fun':check_sum})   # tuple with dictionary inside: check_sum returns 0 if meets reqt
bounds = ((0,1),(0,1),(0,1),(0,1))             # tuple of tuples repeated 4 times: the bounds of each weight to be 0 to 1
bounds = list(zip(np.zeros(10),np.ones(10)))
bounds = list(zip(np.zeros( len(log_ret.columns) ) , np.ones( len(log_ret.columns) )))
# equally weighted
# np.ones(20)*0.05 # 0.05 is 100% divided by 20
initial_guess = [0.25,0.25,0.25,0.25]          # list: initial weights
initial_guess = np.ones(len(log_ret.columns))*(1/len(log_ret.columns))

# Minimizer Function
# minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, 
#          bounds=None, constraints=(), tol=None, callback=None, options=None)
# fun (function to minimize) = neg_sharpe
# x0 = initial_guess
# method = different types of solver algos = 'SLSQP' recommended for basic tasks = Sequential Least SQuares Programming
optimize_results = minimize(neg_sharpe,initial_guess, method = 'SLSQP',bounds=bounds,constraints=constraint)
optimize_results

# Get Optimal Results

# access the "optimize function" results: success or failure?
print('Success: ' + str(optimize_results.success))
# actual values if successful (these are weights)
print('Weights: ' + str(optimize_results.x))

# put weights into function to get Exp Return, Exp Volatility, Sharpe Ratio
print('Results (Exp Return, Exp Volatility, SR): ' + str(get_ret_vol_sr(optimize_results.x)))

# Sharpe Ratio within the Result Set
# seems like maximium sharpe ratio is 1.03 similar to the above which used 5000 random MC guesses (for more securities, optimization is better than MC)
print('Sharpe Ratio: ', get_ret_vol_sr(optimize_results.x)[2])
get_ret_vol_sr(optimize_results.x)[2]



###############################################################################################
# Calculate efficient frontier
# 1) "get highest return for each level of risk" 
# 2) OR "lowest risk possible for given level of return"
# 3) best y value for each x value


# y values storage, 100 points on the 
# look up at the visualization to see which x min and x max (vol min and vol max)
# vol min looks like 0 and vol max looks like 0.3

# NEED TO UPDATE THESE POSSIBLE Y VALUES 
frontier_y = np.linspace(-0.05, 0.4, 50)
# tried 0.46, 0.65, 100
# tried 0, 1, 100
# tried -1, -0.3

# function to return volatility
# give weights
# return the volatility [1]
# comment = we do not need to inverse the volatility, unlike the sharpe ratio 
def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]


frontier_volatility = []

# Calc Start Time
now = datetime.datetime.now()
print("run started: =", now)

# for every possible return on the y axis, what possible x
# eq means equation
# first constraint: all sums have to equal 1 (returns 0)
# second constraint: get the returns and subtract the possible return (that's the max possible return, returns 0)
for possible_return in frontier_y:
    constraint = ({'type':'eq','fun':check_sum},
                  {'type':'eq','fun':lambda w: get_ret_vol_sr(w)[0]-possible_return})
    
    result = minimize(minimize_volatility, initial_guess, method='SLSQP',bounds=bounds, constraints=constraint)
    # fun is function value from result
    frontier_volatility.append(result['fun'])

# Calc End Time
# takes 25 seconds to run (5 minutes for 20 assets?)
print("run ended: =", datetime.datetime.now())
print("run duration: =", datetime.datetime.now()-now)

# Scatter plot out the results from before
fig, ax = plt.subplots(figsize=(12,8))
scatter = ax.scatter(vol_arr, ret_arr,c=sharpe_arr,cmap='plasma')  # cmap='plasma'
# color bar for a fig subplot 
fig.colorbar(scatter, label="Sharpe Ratio")


# Scatter plot the max point, red dot
ax.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')

ax.set_xlabel('Volatility')
ax.set_ylabel('Return')

# Frontier Volatility - Markovitz Portfolio Allocation
# for each desired volatility, what is the maximum return?
# based on historical return, this is what i allocate
ax.plot(frontier_volatility, frontier_y,'g--',linewidth=3)

