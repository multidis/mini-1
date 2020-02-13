import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os


# this is to create the data to use in the plot 
# can be any data
os.chdir("c:\\Users\ebayuser")

# if want to subsample the data
indices = np.arange(0,24999, 10)

x = pd.read_csv("x_coord.csv", ",").to_numpy()
x_min, x_max = min(x) - 0.2*(max(x) - min(x)), max(x) + 0.2*(max(x) - min(x))

y = pd.read_csv("y_coord.csv").to_numpy()
y_min, y_max = min(y) - 0.2*(max(y) - min(y)), max(y) + 0.2*(max(y) - min(y))



# this is to create the actual plots

fig, ax = plt.subplots(figsize=(20,20))
line, = ax.plot(x, y)

# update function
def update(num, x, y, line):
    line.set_data(x[:50*num], y[:50*num])
    line.axes.axis([x_min, x_max, y_min, y_max])
    return line,

ani = animation.FuncAnimation(fig, update, int(len(x)/50), fargs=[x, y, line],
                              interval=25, blit=True)

# save animation to file
ani.save('Matern2.gif')




n = 25000
dx = 1/np.sqrt(n) * np.random.normal(0,1,n)
dy = 1/np.sqrt(n) * np.random.normal(0,1,n)

x = np.cumsum(dx)
y = np.cumsum(dy)

x_min, x_max = min(x) - 0.2*(max(x) - min(x)), max(x) + 0.2*(max(x) - min(x))

y_min, y_max = min(y) - 0.2*(max(y) - min(y)), max(y) + 0.2*(max(y) - min(y))

fig, ax = plt.subplots(figsize=(20,20))
line, = ax.plot(x, y)


# update function
def update(num, x, y, line):
    line.set_data(x[:50*num], y[:50*num])
    line.axes.axis([x_min, x_max, y_min, y_max])
    return line,

ani = animation.FuncAnimation(fig, update, int(len(x)/50), fargs=[x, y, line],
                              interval=25, blit=True)

ani.save('test.gif')