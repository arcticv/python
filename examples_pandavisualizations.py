import numpy as np
import pandas as pd
import seaborn as sns    # makes histogram prettier

import matplotlib.pyplot as plt     # optional color scheme
%matplotlib inline


# This example is to walk through the different types of charts in pandas
# There is a full example below for time series data with numerous axes adjustment codes (dates, save down)



# data load
# remember the index is what they use for x-axis, and the columns are the area/stacks on y-axis 
df1 = pd.read_csv('df1',index_col=0)
df2 = pd.read_csv('df2')
df1.head() # to check the data

'''
df.plot.area
df.plot.barh
df.plot.density
df.plot.hist
df.plot.line
df.plot.scatter
df.plot.bar
df.plot.box
df.plot.hexbin
df.plot.kde
df.plot.pie
'''

# histogram for df1 using column A
df1['A'].hist(bins=30)    
# other variations
df1['A'].plot(kind='hist',bins=30)
df1['A'].plot.hist(bins=30)
# optional color scheme
plt.style.use('dark_background')   
df1['A'].hist()
plt.style.use('ggplot')            
df1['A'].hist()
plt.style.use('bmh')
df1['A'].hist()

# area plot that uses the index along the x-axis (kinda like date)
df2.plot.area(alpha=0.4)

# bar plot
df2.plot.bar()
# stacked bar plot
df2.plot.bar(stacked=True)

# regular line plots (you have to specify X and Y but for Y you can use the column name)
df1.plot.line(x=df1.index,    y='B'     ,    figsize=(12,3),lw=1)

# scatter plot A vs B, then C by color or pass in color map
df1.plot.scatter(x='A', y='B', c='C')
# scatter plot by color map
df1.plot.scatter(x='A',y='B',c='C', cmap='coolwarm')
# scatter plot by size
df1.plot.scatter(x='A', y='B', s=df1['C']*100)

# box plot to plot "distributions by Column"
df2.plot.box() 

# hex heat map
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
# gridsize is size of hexagon
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='Oranges')
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='coolwarm')

# kernel density estimation KDE plot = kinda like a histogram
df2['a'].plot.kde()
df2.plot.density()

# time series visualization
stockdata['Adj. Close'].plot(xlim=['2007-01-01','2012-01-01'], ylim=(20,50))
stockdata['Adj. Close'].plot(xlim=['2007-01-01','2012-01-01'], ylim=(20,50),ls='--',c='red')



#########################################################################################################
# full on example to analyze stock data
# more time series code



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.dates as dates
from matplotlib import pyplot as plt
%matplotlib inline
# if turn on dynamic plots, make sure you restart kernel 
#%matplotlib notebook 

# read data to df1, from yahoo and set parse dates
df1 = pd.read_csv('stock_data.csv',index_col=0,parse_dates=True)
df1.head()
#df1.plot(figsize=(12,8))   # can plot it to see it

# slice df1 to dates, these dates work as you see them
sub_df1 = df1.loc['2020-02-01':'2020-03-26']
sub_df1.head()    
idx = df1.loc['2020-02-01':'2020-03-26'].index
idx
# slice df1 and take one column
stock = df1.loc['2020-02-01':'2020-03-26']['Adj Close']
stock.head()

# plot the data
fig,ax = plt.subplots()
# this is more than a plot, date time indexed information 
ax.plot_date(idx,stock,'-')

# to fix overlapping x-axis
fig.autofmt_xdate()
plt.tight_layout()
# to further enhance x-axis meaning
ax.xaxis.set_major_locator(dates.MonthLocator())
ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n%b-%Y'))   # ax.xaxis.set_major_formatter(dates.DateFormatter('%B-%d-%a'))
plt.xticks(rotation=45)  # rotation='vertical'   
# or plt.xticks(x, labels, rotation='vertical') or plt.xticks(np.arange(3), ['Tom', 'Dick', 'Sue']) 
# turn off ticks
# plt.xticks([], [])
# plt.axis('off')
'''
# for more control:
plt.tick_params(axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=True) # labels along the bottom edge are off
'''

# to add minor x axis
ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=0))  # 0 is monday, 1 is tuesday
ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))   # or %a

# add grid lines
ax.xaxis.grid(True)
ax.yaxis.grid(True)

# save the figure
plt.tight_layout() # prevents the x-axis from being cut off during save
plt.savefig('plot')
