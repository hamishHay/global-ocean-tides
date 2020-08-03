import numpy as np
from numpy import sin as s
from scipy.integrate import simps
from planetPy.constants import AU, G
from planetPy.database import Database
import matplotlib as mpl
import h5py
# import planet_data as data
from planetPy.database import Database
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum
import scipy as sc
from math import isclose
DAY = 24*60.*60.


plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Avante Garde'
plt.rcParams['mathtext.it'] = 'Avante Garde:italic'
plt.rcParams['mathtext.bf'] = 'Avante Garde:bold'
# plt.rc('font',sans_serif="Avante Garde")
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avante Garde']})
# plt.rc('font',serif='Palatino')
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx},\\usepackage{sfmath}')

plt.rc('lines', linewidth=1.0)

plt.rc('figure', dpi=120)

plt.rc('axes', linewidth = 1.2)
# plt.style.use('dark_background')

plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

data = Database()

# i = str(sys.argv[1])
# j = str(sys.argv[2])
# k = str(sys.argv[3])


fig, (ax1) = plt.subplots(nrows=1, figsize=(3.5,3.5), sharex=True)
axes = [ax1]
# print("Mass ratio: ", m1/ms)
# print("Semimajor axis ratio: ", a1/a2)

load_file = h5py.File('../galilean_coefficients.h5','r')

coeff_io = load_file['europa']['io']['tide coeffs'][:, :]
coeff_gan = load_file['europa']['ganymede']['tide coeffs'][:, :]

q_io = np.array(load_file['europa']['ganymede']['frequency'][:], dtype=np.float)
q_gan = np.array(load_file['europa']['ganymede']['frequency'][:], dtype=np.float)

q_gan *= 0.5

FFT_22 = np.zeros(len(q_gan))
FFT_20 = np.zeros(len(q_gan))
for i in range(len(q_gan)):
    if isclose(q_gan[i], q_io[i], abs_tol=0.01):
        FFT_22[i] += coeff_io[i, 4] + coeff_io[i, 11] + coeff_gan[i, 4] + coeff_gan[i, 11]
        FFT_20[i] += coeff_io[i, 0] + coeff_gan[i, 0]
    else:
        FFT_22[i] += coeff_gan[i, 4] + coeff_gan[i, 11]
        FFT_20[i] += coeff_gan[i, 0]

P22_gan =  coeff_gan[:, 4]
P22b_gan =  coeff_gan[:, 11]
P20_gan =  coeff_gan[:, 0]

P22_io =  coeff_io[:, 4]
P22b_io =  coeff_io[:, 11]
P20_io =  coeff_io[:, 0]



#
# for x in range(2):
#     FFT_P22  += tide_coeffs[:, 4]
#     FFT_P22b += tide_coeffs[:, 11]
#     FFT_P20  += tide_coeffs[:, 0]
#
#
#
# # qs  = load_file[i][j]['frequency'][:]
# # qs = np.array(qs, dtype=np.float)
# #
# # # if i is 'europa' and j is 'io':
# # nij = abs(data.moon_dict[i.capitalize()].mean_motion - data.moon_dict[j.capitalize()].mean_motion)
#
#
#
pp = ax1.step(q_gan, abs(P22_gan + P22b_gan),  where='mid', solid_joinstyle='miter', lw=1.0, alpha=0.8, label="$C_{22}$")
pp = ax1.step(q_io, abs(P22_io + P22b_io),  where='mid', solid_joinstyle='miter', lw=1.0, alpha=0.8, label="$C_{22}$")
# pp = ax1.step(q_gan, , where='mid', solid_joinstyle='miter', lw=1.0, alpha=0.8, label="$S_{22}$")
# pp = ax1.step(q_gan, abs(FFT_20),  where='mid', solid_joinstyle='miter', lw=1.0, alpha=0.8, label="$C_{20}$")

ax1.legend(frameon=False)

for ax in axes:
    ax.set_xlim([-20,20])
    # ax.set_yscale("log")
    # ax.set_ylim([1e-3, 1.01])


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
    # ax.grid(which='major', lw=0.5, alpha=0.5)
    # ax.grid(which='minor', lw=0.3, alpha=0.3)
# ax2.set_xlim([0, 40])
# ax1.legend(fontsize=9)

ax1.set_ylabel("Normalised tidal forcing fourier coefficients", fontsize=9)
# ax2.set_ylabel("Normalised Fourier coefficient,\n$|a_{221q}|$", fontsize=10)
# ax2.set_ylabel("$A_{221q}$")
ax1.set_xlabel("Frequency, $q$", fontsize=9)
# ax2.set_xlabel("Frequency, $q$")

# ax1.text(11, 0.9,"Tides on TRAPPIST-1 "+planets[i]+" by "+planets[j], fontsize=9)


# fig.suptitle("Frequency spectrum of the normalised\ntidal forcing Fourier coefficients")
# plt.subplots_adjust(hspace=0.04)
# ax2.legend()
plt.show()
fig.savefig("/home/hamish/Dropbox/Tests/forcing_spectrum.pdf", dpi=600, bbox_inches='tight', transparent=False)
