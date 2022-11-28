#!/usr/bin/env python3
############################
#
# template by M Joyce
# for use with Smith College students
#
############################


################################
#
# cell 1
#
################################
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy import stats
from scipy.stats import norm

#print('modules imported')

################################
#
# cell 2
#
################################
def set_fig(ax):
    ax.tick_params(axis = 'both',which='both', width=2)
    ax.tick_params(axis = 'both',which='major', length=12)
    ax.tick_params(axis = 'both',which='minor', length=8, color='black')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    return 

#print("plot settings function defined")


################################
#
# cell/step 3
#
################################
data_file = 'stellar_ages.dat'
Joyce_ages= np.loadtxt(data_file, usecols=(0), unpack = True)

(Jmu, Jsigma) = norm.fit(Joyce_ages)
Jstats=r'$\mu=$' + "%.2f"%Jmu + ';'+r' $\sigma=$' + "%.2f"%Jsigma


#print("ages successfully read in")


################################
#
# cell/step 4
#
################################
fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)

n, bins, patches = plt.hist(Joyce_ages,  bins="auto", alpha = 1, color= 'navy')
y = scipy.stats.norm.pdf(bins, Jmu, Jsigma)*len(Joyce_ages)
plt.plot(bins, y, '--', color='cornflowerblue', linewidth=5, label='normal distribution: '+Jstats)

plt.xlabel('Ages (Gyr)', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.legend(loc=2)
plt.show()
plt.close()


################################
#
# cell/step 5
#
################################
idx_array = np.linspace(min(Joyce_ages), max(Joyce_ages), 1000)
kde = stats.gaussian_kde(Joyce_ages)
plt.plot(idx_array, kde(idx_array)*len(Joyce_ages), linewidth=5, linestyle='-', color='lightblue', label='KDE')

plt.xlabel('Ages (Gyr)', fontsize=16)
plt.ylabel('Kernel Density Estimate', fontsize=16)
plt.show()
plt.close()


##################################
# EXERCISE
##################################
# Using the above as a template, compare a Gaussian 
# versus KDE fit to the "Bensby Ages" 
# in stellar_ages.dat
#

Bensby_ages = ...
(..., ...) = norm.fit(Bensby_ages)
Bstats= ...

## hint: repeat steps 3, 4, 5 with new variable names


################################
#
# cell 6
#
################################
fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)

plt.hist(Joyce_ages,  bins="auto", alpha = 1, color= 'navy')
plt.plot(bins, y, '--', color='cornflowerblue', linewidth=5, label='normal distribution: '+Jstats)
plt.plot(idx_array, kde(idx_array)*len(Joyce_ages), linewidth=5, linestyle='-', color='lightblue', label='KDE')

plt.xlabel('Ages (Gyr)', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.legend(loc=2)

plt.show()
plt.close()

################################
#
# cell 7
#
################################
########################################
#
# resampling 
#
########################################
import sklearn
from sklearn.neighbors import KernelDensity

kde1 = KernelDensity(bandwidth=1).fit(Joyce_ages.reshape(-1, 1))
y1 = kde1.sample(91)

fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)
ax.hist(y1, bins=8, density=False, color='lightblue', alpha=0.5, label=r'$h = 1$')

plt.show()
plt.close()

##################################
# EXERCISE
##################################
# repeat the above, setting the bandwidth to 0.1 and 0.01
# plot all three kdes on top of each other
# which is the best fit to the Joyce Ages histogram?


##################################
# EXERCISE
##################################
# Make a sklearn KDE for Bensby's ages
# Visually, determine an appropriate value for the bandwidth parameter in this case
