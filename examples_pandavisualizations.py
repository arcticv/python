import numpy as np
import pandas as pd
import seaborn as sns    # makes histogram prettier

import matplotlib.pyplot as plt     # optional color scheme

%matplotlib inline


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

