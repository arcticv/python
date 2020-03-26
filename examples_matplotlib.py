# https://github.com/rougier/matplotlib-tutorial

# call the figure
# iterate on each and plot

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = np.arange(0,100)
y = x*2
z = x**2

# Example 1 (regular plot)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_title('hi')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.show()

# Example 2 (zoomed second plot within first plot, on top of each other)
fig = plt.figure()    # create a figure object
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,0.4,0.4]) # x location, y location, x length, y width
ax1.plot(x,y,color='blue')
ax2.plot(x,y,color='blue') # insert plot
ax2.set_title('zoom')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim([20,25])
ax2.set_ylim([30,50])

# Example 3 (2 subplots side by side)
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,3))
axes[0].plot(x,y,color='blue',ls='--',lw=2)
axes[1].plot(x,z,color='red',lw=3)


