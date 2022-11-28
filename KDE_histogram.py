#!/usr/bin/env python3
import numpy as np
#import stats

import sklearn
from sklearn.neighbors import KernelDensity


import matplotlib.pyplot as plt
import sys
import subprocess
import scipy
from scipy import stats
from scipy.stats import norm
sys.path.append('../../../MESA/pyMESA/')
sys.path.append('../../../bulge_isochrones/')


# import isochrone_module as im 		# mine
import photometry_module as ph 		# mine

##########
bin_num = 'auto'
save_name = 'KDE_histogram.pdf'
###########



data_file = 'stellar_ages.dat'
Joyce_ages, Bensby_ages= np.loadtxt(data_file, usecols=(0,1), unpack = True)


(Jmu, Jsigma) = norm.fit(Joyce_ages)
(Bmu, Bsigma) = norm.fit(Bensby_ages)

Jstats=r'$\mu=$' + "%.2f"%Jmu + ';'+r' $\sigma=$' + "%.2f"%Jsigma
Bstats=r'$\mu=$' + "%.2f"%Bmu + ';'+r' $\sigma=$' + "%.2f"%Bsigma


#np.histogram()
fig, ax = plt.subplots(figsize = (16,16))
ph.set_fig(ax)


n, bins, patches = plt.hist(Joyce_ages,  bins=bin_num, alpha = 1, color= 'navy',\
						    label = 'current work: MIST ages\nover basis of 0.5-20 Gyr')#+'\n'+Jstats)
#						    edgecolor = 'cornflowerblue', linewidth=5, linestyle = ':',\

y = scipy.stats.norm.pdf(bins, Jmu, Jsigma)*len(Joyce_ages)
plt.plot(bins, y, '--', color='cornflowerblue', linewidth=5, label='normal distribution: '+Jstats)


#X = Joyce_ages
#idx_array = np.arange(0, np.floor(max(Joyce_ages)), 1)

#idx_array = np.arange(0, len(Joyce_ages), 1)
idx_array = np.linspace(min(Joyce_ages), max(Joyce_ages), 1000)
print('idx_array: ',idx_array)

#X = [idx_array, Joyce_ages]

#bandwidth = 2 ## but actually load age uncertainties for real exercise
#kde  = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth ).fit(X)

#weights = [] ## same size as data array
#kde = stats.gaussian_kde(Joyce_ages, bw_method = 1)
kde = stats.gaussian_kde(Joyce_ages)

#print("kde: ", kde
ax.plot(idx_array, kde(idx_array)*len(Joyce_ages), linewidth=5, linestyle='-', color='lightblue', label='KDE')


# #							edgecolor = 'orange', linewidth=5, linestyle = ':',\
# n, bins, patches = plt.hist(Bensby_ages,bins = bin_num, alpha = 0.65, color='red',\
# 						    label = 'Ages from Bensby et al.')#+'\n'+Bstats)
# y = scipy.stats.norm.pdf(bins, Bmu, Bsigma)*len(Bensby_ages)
# plt.plot(bins, y, ':', color='orange', linewidth=5, label='Bensby et al. '+Bstats)


plt.xlabel('Ages (Gyr)', fontsize=30)
plt.ylabel('Count', fontsize=30)


text_str = 'For ' + str(len(Joyce_ages) ) + ' targets of Bensby et al.\nAges from Teff, logg'

ax.annotate(text_str, xy=(1, 1), xytext=(-15, -15), fontsize=14,
xycoords='axes fraction', textcoords='offset points',
bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.65),\
horizontalalignment='right', verticalalignment='top') 

plt.legend(loc=2, fontsize=14)
plt.savefig(save_name)

#plt.show()

plt.close()
subprocess.call('pdfcrop '+ save_name + ' ' + save_name, shell = True)