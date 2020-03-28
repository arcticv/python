########################################################################################################



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

# check yearly sample mean
df.resample(rule='A').mean()

