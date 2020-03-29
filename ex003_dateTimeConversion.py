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
