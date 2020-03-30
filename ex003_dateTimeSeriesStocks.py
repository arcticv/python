########################################################################################################
# modifying date columns

# check if string or datetime
df.info()
# passing in entire series and will be reformatted to a datetime object from standard notation yyyy-mm-dd
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].apply(pd.to_datetime) # same as above
# if not standard notation: raw_data['Mycol'] =  pd.to_datetime(raw_data['Mycol'], format='%d%b%Y:%H:%M:%S.%f')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.info()
# move date to index
df.set_index('Date',inplace=True)
df.head()

# or you can do it all in one step, providing that the dates are clean
df = pd.read_csv('time_data/walmart_stock.csv', index_col='Date', parse_dates = True)

########################################################################################################
# "group by" using the resample method

# calculate the average = yearly sample mean
df.resample(rule='A').mean()    # annual resampling
df.resample(rule='Q').mean()    # quarterly resampling
df.resample(rule='BQ').mean()   # business quarterly resampling (on a Friday)
# other rules max, mean, std, or a lambda expression
df.resample(rule='A').max()     # maximum within the annual resampling
# create custom resampling function
def first_day(entry):
  return entry[0]
# pass it in
df.resample('A').apply(first_day)

# chart out the mean for the year, not a continuous plot
df['Close'].resample('M').mean().plot(kind='bar',figsize=(12,5))


########################################################################################################
# shifting the data

# Shift the data to the next row, first row is NaN, and you will lose that last piece of data that no longer has an index
df.shift(1).head()
# Shift the data up a row, last row is NaN
df.shift(-1).tail()

# Shift the index itself to match up frequency provided - good for pivot tables
df.tshift(periods=1,freq='M').head() # index 2012-01-03 becomes 2012-01-31, 2012-01-04 becomes 2012-01-31
df.tshift(periods=1,freq='Y').head() # index 2012-01-03 becomes 2012-12-31, 2012-01-04 becomes 2012-12-31

########################################################################################################
# rolling average mean = moving average = general trend when noisy (set time period, average statistic)
df = pd.read_csv('walmart_stock.csv', index_col='Date',parse_dates=True)
df['Open'].plot(figsize=(12,6))
df.rolling(window=7).mean().head(20) # first 6 rows are NaN
# plotting with legend
df['Close 7D MA'] = df['Close'].rolling(window=7).mean()
df['Close 30D MA'] = df['Close'].rolling(window=30).mean()
df[['Close', 'Close 30D MA','Close 7D MA']].plot(figsize=(16,6)) 

########################################################################################################
# bollinger bands to signal and identify tops and bottoms
#
# plotting with legend
df['Close 7D MA'] = df['Close'].rolling(window=7).mean()
df['Close 30D MA'] = df['Close'].rolling(window=30).mean()
df[['Close', 'Close 30D MA','Close 7D MA']].plot(figsize=(16,6)) 

# Close 20 MA
df['Close 20D Mean'] = df['Close'].rolling(20).mean()

# Upper = 20MA + 2*std(20)
df['Upper 20D'] = df['Close 20D Mean'] + 2*(df['Close'].rolling(20).std())

# Lower = 20MA - 2*std(20)
df['Lower 20D'] = df['Close 20D Mean'] - 2*(df['Close'].rolling(20).std())

# Plot 1 - with the bands
df[['Close', 'Upper 20D','Lower 20D']].plot(figsize=(16,6)) 
# Plot 2 - zoomed in
df[['Close', 'Upper 20D','Lower 20D']].tail(200).plot(figsize=(16,6)) 


########################################################################################################
# import stock data using datareader and plot with legend
#
import pandas_datareader.data as web 
import datetime
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2017, 1, 1)
df1 = web.DataReader('TSLA', 'yahoo', start, end)
df2 = web.DataReader('F', 'yahoo', start, end)
df3 = web.DataReader('GM', 'yahoo', start, end)
#loop formula if you want it
#symbols = ['AAPL', 'MSFT', 'AABA', 'DB', 'GLD']
#webData = pd.DataFrame()
#for stockSymbol in symbols:
#    webData[stockSymbol] = web.DataReader(stockSymbol, 
#    data_source='yahoo',start= 
#               startDate, end= endDate, retry_count= 10)['Adj Close']   
#    time.sleep(22) # thread sleep for 22 seconds.
#
# Some column renaming if you need it
df1['TSLA Close'] = df1['Close']
df2['Ford Close'] = df2['Close']
df3['GM Close'] = df3['Close']
# Plot them
df1['TSLA Close'].plot(label='Tesla', title='Closing Prices', figsize=(15,5))
df2['Ford Close'].plot(label='Ford')
df3['GM Close'].plot(label='GM')
# how do you set a label and legend!?!?!
plt.legend()
plt.show()
# plot the volume
df1['Volume'].plot(label='Tesla', title='Closing Prices', figsize=(15,5))
df2['Volume'].plot(label='Ford')
df3['Volume'].plot(label='GM')
plt.legend()
plt.show()
# find the max volume and max volume date
df2['Volume'].max()
df2['Volume'].idxmax() # index max for time stamp
# now, plot the moving average 
df2['Open'].plot(label='GM Open',figsize=(16,8))
df2['Open'].rolling(50).mean().plot(label='GM MA50')
df2['Open'].rolling(200).mean().plot(label='GM MA200')
plt.legend()

# moving average and standard deviation filled in areas
df2['Open'].plot(label='GM Open',figsize=(16,8))
# plt.fill_between(mstd.index, ma - 2 * mstd, ma + 2 * mstd, color='b', alpha=0.2)
mavg = df2['Open'].rolling(20).mean()
mstd = df2['Open'].rolling(20).std()
plt.fill_between(mstd.index, mavg - 2 * mstd, mavg + 2 * mstd, color='gray', alpha=0.2)

# for changes you can do a cumulative sum
dfDeltas = dfDeltas.cumsum()
# neat way of creating the x axis (dates) dynamically:
df3['A'] = pd.Series(list(range(len(df))))
df3.plot(x='A', y=dfDeltas)

########################################################################################################
# scatter matrix 
# source https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#visualization-scatter-matrix
# build a scatter plot to see how car companies relate
from pandas.plotting import scatter_matrix 
# build a single dataframe by joining the columns together
cars_df = pd.concat([df1['Open'],df2['Open'],df3['Open']],axis=1)
# add column labels instead of just 3 Opens
cars_df.columns = ['Tesla Open', 'GM Open', 'Ford Open']
cars_df.head()
# this is the scatter matrix to see correlations
# since so many points, use alpha to see more points, and also add more histogram binds
scatter_matrix(cars_df,figsize=(8,8), alpha=0.2, hist_kwds={'bins':50})

########################################################################################################
# red and green candle chart 
# source https://matplotlib.org/examples/pylab_examples/finance_demo.html
# build a scatter plot to see how car companies relate

# need to install matplotlib.finance: pip install mpl_finance
# deprecated: from matplotlib.finance import candlestick_ohlc
# does not work: from mplfinance import candlestick_ohlc
from mpl_finance import candlestick_ohlc
# ohlc = (open, high, low, close) in tuples
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY

# grab everything in January then reset the index
ford_reset = df3.loc['2012-01'].reset_index()
ford_reset.head()
ford_reset.info()
# create a numerical value for a time series index because matplotlib is not good at date processing
ford_reset['date_ax'] = ford_reset['Date'].apply(lambda date: date2num(date))
ford_reset.head()
# create columns for OHLC
list_of_cols = ['date_ax','Open','High','Low','Close']
# create an array of tuple values
ford_values = [tuple(vals) for vals in ford_reset[list_of_cols].values ]
ford_values
# copy and paste the date formatter code
mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12
# Plot function
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
#ax.xaxis.set_minor_formatter(dayFormatter)
candlestick_ohlc(ax, ford_values, width=0.3, colorup='g', colordown='r');
