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
# cell 3
#
################################
data_file = 'stellar_ages.dat'
Joyce_ages, Bensby_ages= np.loadtxt(data_file, usecols=(0,1), unpack = True)

(Jmu, Jsigma) = norm.fit(Joyce_ages)
(Bmu, Bsigma) = norm.fit(Bensby_ages)

Jstats=r'$\mu=$' + "%.2f"%Jmu + ';'+r' $\sigma=$' + "%.2f"%Jsigma
Bstats=r'$\mu=$' + "%.2f"%Bmu + ';'+r' $\sigma=$' + "%.2f"%Bsigma

#print("ages successfully read in")



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
ax.hist(y1,          bins=8, density=False, color='lightblue', alpha=0.5, label=r'distribution drawn from $h = 1$')

plt.show()
plt.close()

################################
#
# cell 8
#
################################
kde2 = KernelDensity(bandwidth=0.1).fit(Joyce_ages.reshape(-1, 1))
y2 = kde2.sample(91)

fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)
ax.hist(y2, 		 bins=8, density=False, color='purple',    alpha=0.7, label=r'distribution drawn from $h = 0.1$')


plt.show()
plt.close()

################################
#
# cell 9
#
################################
kde3 = KernelDensity(bandwidth=0.01).fit(Joyce_ages.reshape(-1, 1))
y3 = kde3.sample(91)

fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)
ax.hist(y3, 		 bins=8, density=False, color='magenta',   alpha=0.7, label=r'distribution drawn from $h = 0.01$')

plt.show()
plt.close()

################################
#
# cell 10
#
################################
fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)

#ax.plot(idx_array, y1, color='lightblue', label='KDE 1')
#ax.hist(y1_resample, bins=8, density=False, color='lightblue', alpha=0.5, label=r'KDE 1 resampled; $h=auto$') ## set density = False
ax.hist(y1,          bins=8, density=False, color='lightblue', alpha=0.5, label=r'$h = 1$')
ax.hist(y2, 		 bins=8, density=False, color='purple',    alpha=0.7, label=r'$h = 0.1$')
ax.hist(y3, 		 bins=8, density=False, color='magenta',   alpha=0.7, label=r'$h = 0.01$')
plt.legend(loc=2)

plt.show()
plt.close()