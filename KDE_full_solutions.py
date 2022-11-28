#!/usr/bin/env python3
############################
#
# template by M Joyce
# for use with Smith College students
#
############################


################################
#
# cell/step 1
#
################################
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
from scipy import stats
from scipy.stats import norm

#print('modules imported')

################################
#
# cell/step 2
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

#n, bins, patches = plt.hist(Joyce_ages,  bins="auto", alpha = 1, color= 'navy')
nbins = 8
gaussian_pdf = scipy.stats.norm.pdf(nbins, Jmu, Jsigma)*len(Joyce_ages)
plt.plot(nbins, gaussian_pdf, '--', color='cornflowerblue', linewidth=5, label='normal distribution: '+Jstats)

plt.xlabel('Ages (Gyr)', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.legend(loc=2)
plt.show()
plt.close()

#sys.exit()

################################
#
# cell/step 5
#
################################
idx_array = np.linspace(min(Joyce_ages), max(Joyce_ages), 1000)
kde = stats.gaussian_kde(Joyce_ages)

#sys.exit()

################################
#
# cell/step 6
#
################################
fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)

plt.hist(Joyce_ages,  bins="auto", alpha = 1, color= 'navy')
plt.plot(nbins, gaussian_pdf, '--', color='cornflowerblue', linewidth=5, label='normal distribution: '+Jstats)
plt.plot(idx_array, kde(idx_array)*len(Joyce_ages), linewidth=5, linestyle='-', color='lightblue', label='KDE')

plt.xlabel('Ages (Gyr)', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.legend(loc=2)
plt.show()
plt.close()

#sys.exit()

##################################
# EXERCISE 1
##################################
# Using the above as a template, compare a Gaussian 
# versus KDE fit to the "Bensby Ages" 
# in stellar_ages.dat
#
## hint: repeat steps 3, 4, 5 and 6 with NEW variable names
#
# Ex):
#
#    Bensby_ages = ...
#    (..., ...) = norm.fit(Bensby_ages)
#    Bstats= ...


################################
# SOLUTION 1: cell/step 3 for Bensby
################################
Bensby_ages= np.loadtxt(data_file, usecols=(1), unpack = True)
(Bmu, Bsigma) = norm.fit(Bensby_ages)
Bstats=r'$\mu=$' + "%.2f"%Bmu + ';'+r' $\sigma=$' + "%.2f"%Bsigma

################################
# SOLUTION 1: cell/step 4 for Bensby
################################
fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)

nbins = 8
gaussian_pdf2 = scipy.stats.norm.pdf(nbins, Bmu, Bsigma)*len(Bensby_ages)

plt.plot(nbins, gaussian_pdf2, '--', color='cornflowerblue', linewidth=5, label='normal distribution: '+Bstats)

plt.xlabel('Ages (Gyr)', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.legend(loc=2)
plt.show()
plt.close()

################################
# SOLUTION 1: cell/step 5 for Bensby
################################
idx_array_Bensby = np.linspace(min(Bensby_ages), max(Bensby_ages), 1000)
kde_Bensby = stats.gaussian_kde(Bensby_ages)

################################
# SOLUTION 2: cell/step 6 for Bensby
################################
fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)

plt.hist(Bensby_ages,  bins="auto", alpha = 1, color= 'navy')
plt.plot(nbins, gaussian_pdf2, '--', color='cornflowerblue', linewidth=5, label='normal distribution: '+Bstats)
plt.plot(idx_array_Bensby, kde_Bensby(idx_array_Bensby)*len(Bensby_ages), linewidth=5, linestyle='-', color='lightblue', label='KDE')

plt.xlabel('Ages (Gyr)', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.legend(loc=2)
plt.show()
plt.close()

sys.exit()

################################
#
# cell/step 7 -- Resampling the KDE
#
# you will need to comment out  
#          sys.exit() 
# above to proceed
#
################################
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
# EXERCISE 2
##################################
# repeat the above, setting the bandwidth to 0.1 and 0.01
# plot all three kdes on top of each other
# which is the best fit to the Joyce Ages histogram?


##################################
# EXERCISE 3
##################################
# Make a sklearn KDE for Bensby's ages
# Visually, determine an appropriate value for the bandwidth parameter in this case




################################
#
# cell 8
#
################################
kde2 = KernelDensity(bandwidth=0.1).fit(Joyce_ages.reshape(-1, 1))
gaussian_pdf2 = kde2.sample(91)

fig, ax = plt.subplots(figsize = (8,8))
set_fig(ax)
ax.hist(gaussian_pdf2, 		 bins=8, density=False, color='purple',    alpha=0.7, label=r'distribution drawn from $h = 0.1$')


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
ax.hist(gaussian_pdf2, 		 bins=8, density=False, color='purple',    alpha=0.7, label=r'$h = 0.1$')
ax.hist(y3, 		 bins=8, density=False, color='magenta',   alpha=0.7, label=r'$h = 0.01$')
plt.legend(loc=2)

plt.show()
plt.close()