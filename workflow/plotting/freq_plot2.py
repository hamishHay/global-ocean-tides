import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.misc import factorial

import matplotlib.gridspec as gridspec


fmax=25



data = h5py.File("../../LTE_solutions_alpha_1e-11.h5", 'r')

phi = data['europa']['io']['velocity potential']
psi = data['europa']['io']['stream function']
ho  = data['europa']['io']['ocean thickness']


n_max = len(phi[0, 0, 0, :])
n = np.arange(1, n_max+1, 1)

Nnm0 = n*(n+1)/(2*n+1)*factorial(n)/factorial(n)
Nnm2 = n*(n+1)/(2*n+1)*factorial(n+2)/factorial(n-2)
Nnm1 = n*(n+1)/(2*n+1)*factorial(n+1)/factorial(n-1)
Nnm2[0] = 1.0

freqs = np.arange(-100, 100, 1)

data1 = np.zeros((len(ho), len(freqs)))
for x in range(len(ho)):
    data1[x, :] += np.sum(Nnm0 * (abs(phi[x,:,0,:])**2.0 + abs(psi[x,:,0,:])**2.0), axis=-1)
    data1[x, :] += np.sum(Nnm2 * (abs(phi[x,:,2,:])**2.0 + abs(psi[x,:,2,:])**2.0), axis=-1)
    data1[x, :] += np.sum(Nnm1 * (abs(phi[x,:,1,:])**2.0 + abs(psi[x,:,1,:])**2.0), axis=-1)

data1 = np.log10(data1)
data1[data1 > 15] = 15
data1 = np.ma.masked_where(data1 < 3, data1)




gs = gridspec.GridSpec(2, 3)
# fig, ax = plt.subplots()
# ax.contourf(ho, freqs, np.log10(data1.T))

ax1 = plt.subplot(gs[:,:2])
ax2 = plt.subplot(gs[:, 2])

pp = ax1.pcolormesh(ho, freqs-0.5, data1.T, cmap=plt.cm.plasma, vmin=3)
# ax1.contourf(ho, freqs, data1.T, cmap=plt.cm.plasma)
# ax.loglog(ho, data1)

cb = plt.colorbar(pp, pad=0.3)
cb.set_label("log$_{10}$(Dissipated Power [W])")


# print(psi[:,:,:,:])

ax1.set_xscale("log")

ax1.set_ylim([-fmax,fmax])




load_file = h5py.File('../galilean_coefficients.h5','r')

tide_coeffs = np.array(load_file['europa']['io']['tide coeffs'][:, :])
qs = np.arange(0, len(tide_coeffs[:,0]))



FFT_P22 =  tide_coeffs[:, 4]
FFT_P22b =  tide_coeffs[:, 11]
FFT_P20 =  tide_coeffs[:, 0]

FFT_22 = abs(FFT_P22 + FFT_P22b)
FFT_20 = abs(FFT_P20)

# max_c = max(np.amax(FFT_P20), np.amax(FFT_P22))
max_c = 1

# ax1.fill_between(qs,FFT/max_c, step="post", alpha=0.3)
# ax1.fill_between(qs,FFT_P22/max_c, step="post", alpha=0.15)
# ax1.fill_between(qs,FFT_P22b/max_c, step="post", alpha=0.15)

# ax1.axvline(abs(ns[i]/(ns[i]-ns[j])), lw=1.0, ls='--', color='w')
pp = ax2.step(FFT_22/max_c, freqs,  where='mid', solid_joinstyle='miter', lw=1.0, alpha=0.8, label="$m=2$")
pp = ax2.step(FFT_20/max_c, freqs,  where='mid', solid_joinstyle='miter', lw=1.0, alpha=0.8, label="$m=0$")
# pp = ax1.step(qs, FFT_P22b/max_c, where='mid', solid_joinstyle='miter', lw=1.0, alpha=0.8, label="$S_{22}$")
# pp = ax1.step(qs, FFT_P20/max_c,  where='mid', solid_joinstyle='miter', lw=1.0, alpha=0.8, label="$C_{20}$")

ax2.set_ylim([-fmax,fmax])

ax2.legend(frameon=False, prop={'size': 8})



# nij = 2.05e-5
# a=ax2.get_yticks().tolist()
# for i in range(len(a)):
#     q = int(a[i])
#     if q != 0:
#         a[i] = str( (2*np.pi/(q*nij))/(60*60) )
#
# ax2.set_yticklabels(a)


ax1.set_ylabel("Frequency, $q$")
ax1.set_xlabel("Ocean thickness [km]")
ax2.yaxis.tick_right()

# ax2.set_xscale("log")
# ax2.set_xlim([1e-3, 0.15])



fig = plt.gcf()
fig.savefig("/home/hamish/Dropbox/Tests/freqs.png", dpi=600, bbox_inches='tight')
