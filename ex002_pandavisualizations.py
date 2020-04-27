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
df1.info() # to check the data
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

#########################################################################################################
# slicing examples
df3[['A','B']].plot.box()  # only columns A and B
df3.iloc[0:30].plot.area()    # area plot, slicing only first 30 rows
sub_df1 = df1.loc['2020-02-01':'2020-03-26']  # slice df1 to dates, these dates work as you see them
# time series visualization - slicing within the plot
stockdata['Adj. Close'].plot(xlim=['2007-01-01','2012-01-01'], ylim=(20,50))
stockdata['Adj. Close'].plot(xlim=['2007-01-01','2012-01-01'], ylim=(20,50),ls='--',c='red')
# zoom in to a date range and plot 2 columns
df[['realgdp','trend']]['2000-03-31':].plot()


# some datetime magic
idx = df1.iloc[[0,10]].index
df1.iloc[[0,10]].plot.area()

# specific dates selection - need datetime conversion = this does not work: stock = df1.loc['2020-02-01','2020-03-26']['Adj Close']
df1.loc[[datetime(2020,2,1),datetime(2020,3,26)]].plot.area()  
# the transpose fixes the plot (plot columns on x axis and use index as y axis using pandas)
mydata.loc[[datetime(2020,3,20),datetime(2020,3,27)]].T.plot()
mydata.loc['2020-03-20':'2020-03-27'].plot()
mydata.loc[[datetime.strptime('2020-03-20', '%Y-%m-%d'),datetime.strptime('2020-03-27', '%Y-%m-%d')]].T.plot()

#########################################################################################################
# histogram for df1 using column A
df1['A'].hist(bins=30)    
df3['a'].plot.hist(edgecolor='black', bins=30)
# other variations
df1['A'].plot(kind='hist',bins=30)
df1['A'].plot.hist(bins=30)
# optional color scheme
plt.style.use('dark_background')   
df1['A'].hist()
plt.style.use('ggplot')            
df3['A'].plot.hist(edgecolor='black', bins=30)
plt.style.use('bmh')
df1['A'].hist()


#########################################################################################################
# area plot that uses the index along the x-axis (kinda like date)
df2.plot.area(alpha=0.4)
# area plot, slicing only first 30 rows
df3.iloc[0:30].plot.area()
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))   # put the legend on the outside of the box
plt.show()


#########################################################################################################
# bar plot
df2.plot.bar()
# stacked bar plot
df2.plot.bar(stacked=True)


#########################################################################################################
# regular line plots (you have to specify X and Y but for Y you can use the column name)
df1.plot.line(x=df1.index,    y='B'     ,    figsize=(12,3),lw=1)

#########################################################################################################
# scatter plot A vs B, then C by color or pass in color map
df1.plot.scatter(x='A', y='B', c='C')
# scatter plot by color map
df1.plot.scatter(x='A',y='B',c='C', cmap='coolwarm')
# scatter plot by size
df1.plot.scatter(x='A', y='B', s=df1['C']*100)
# scatter plot 
df3.plot.scatter(x='a',y='b',c='red',s=50,figsize=(12,3))   # s is for size


#########################################################################################################
# box plot to plot "distributions by Column"
df2.plot.box()  # all columns
df3[['A','B']].plot.box()  # only columns A and B


#########################################################################################################
# hex heat map
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
# gridsize is size of hexagon
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='Oranges')
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='coolwarm')


#########################################################################################################
# kernel density estimation KDE plot = kinda like a histogram
df2['a'].plot.kde()
df2.plot.density()

# kde with shaded area underneath
import seaborn as sns 
# List of five companies to plot
columns = ['a', 'b']
# Iterate through the companies
for column in columns:
    # Subset the dataframe to only the company
    # subset = df3[df3['name'] == column]
    subset = df3[column]
    # Draw the density plot
    # sns.distplot(subset['arr_delay'], hist = False, kde = True, kde_kws = {'linewidth': 3}, label = airline)
    sns.distplot(subset, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3},
                 label = column)    
# Plot Formatting
plt.legend(prop={'size': 14}, title = 'Company')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))   # move the legend outside
plt.title('Density Plot with Multiple Companies')
plt.xlabel('Price Change (USD)')
plt.ylabel('Density')


#########################################################################################################
# Seaborn Count Plot with sorted order along x axis
plt.figure(figsize=(12,5))
# create an order array for seaborn countplot
sub_grade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order=sub_grade_order, palette='coolwarm')


#########################################################################################################
# Seaborn Heatmap with labels
# Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True, cmap='viridis')
plt.ylim(10,0)
# notice loan amount and installment have almost perfect correlation

#########################################################################################################
# Seaborn Scatter Plot Example with color map options and linear regression lines

# a plot to see graduate tuitions vs room and board
# scatter plot in seaborns uses lmplot
sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', scatter=True,
          fit_reg=True, palette='prism',  # coolwarm Blues
          height=6, aspect=1, 
          legend_out=True, scatter_kws={"s": 10})

#########################################################################################################
# seaborn hist histogram using facetgrid and map
sns.set_style("whitegrid", {'axes.grid' : False,'axes.edgecolor':'none'})
fg = sns.FacetGrid(df, hue='PrivateSchoolYN', height=6, aspect=2, palette='coolwarm')
fg = fg.map(plt.hist,'OutstateTuitionAmount',bins=20,alpha=0.7).add_legend()


#########################################################################################################
# Seaborn Scatter Plot Example with color map options and linear regression lines
""" 
Colormap possible values are: 
Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, 
Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, 
PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, 
PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, 
RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, 
Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, 
YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, 
binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, 
coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, 
gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, 
gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, 
hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, 
nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, 
rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, 
tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, 
twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r
"""




#########################################################################################################
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
df1.info()
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
