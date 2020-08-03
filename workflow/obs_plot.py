import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.misc import factorial
from pyshtools.expand import MakeGridDH
import pyshtools
import string
from matplotlib import font_manager

from matplotlib.ticker import MultipleLocator, LogLocator

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx}, \\usepackage{libertine}, \\usepackage{libertinust1math}, \\sisetup{detect-all}')

plt.rc('lines', linewidth=0.6)

# plt.rc('pdf.fonttype', fonttype=42)
mpl.rcParams['pdf.fonttype'] = 42

# plt.rcParams['figure.figsize'] = (7,4)

labelsize=9
ticksize=8

p_coeffs = h5py.File("galilean_coefficients.h5", 'r')
ecc_coeff = p_coeffs['europa']['eccentricity']['tide coeffs'][:]
io_coeff = p_coeffs['europa']['io']['tide coeffs'][:]
gan_coeff = p_coeffs['europa']['ganymede']['tide coeffs'][:]

p_coeffs = (ecc_coeff + io_coeff + gan_coeff)*0.5

# p_coeffs = np.concatenate( (p_coeffs[40:100], p_coeffs[101:-39]), axis=0 )
p_coeffs = np.concatenate( (p_coeffs[100-80:100], p_coeffs[101:-(100-80-1)]), axis=0 )
# p_coeffs = np.concatenate( (p_coeffs[:100], p_coeffs[101:]), axis=0 )

# p_coeffs = p_coeffs[:-40]

# print(p_coeffs[:,0])
U20 = p_coeffs[:, 0]
U22_C = p_coeffs[:, 4]
U22_S = p_coeffs[:, 11]

love_file = h5py.File("shell_love_numbers.h5", 'r')
beta = love_file['europa']["beta"][10 - 1][1]
ups = love_file['europa']["upsilon"][10 - 1][1]

g = 1.3079460990117284

p_coeffs *= ups/beta * 1./g

r = 1560.8e3
force_freq = 2.05e-5*0.5

# hsim = 73.2186e3 - 0.1
hsim = 73.2181e3

hres = 73218.06615773
hsim = hres
hsim = 71e3
# hsim = 4e3
# hsim =73.0e3

target_h = [80e3, 50e3]

# file_names = ["/home/hamish/Research/LaplaceTidalAnalytic/workflow/europa_disp_ecc_io_h80km.h5",
#               "/home/hamish/Research/LaplaceTidalAnalytic/workflow/europa_disp_ecc_io_h50km.h5"]

file_names = ["/home/hamish/Research/LaplaceTidalAnalytic/workflow/europa_v1e19_a1e-11.h5",
              "/home/hamish/Research/LaplaceTidalAnalytic/workflow/europa_v1e19_a1e-11.h5"]

file_names = ["/home/hamish/Research/LaplaceTidalAnalytic/workflow/europa_obs_v1e17_a1e-11.h5",
              "/home/hamish/Research/LaplaceTidalAnalytic/workflow/europa_obs_v1e17_a1e-11.h5"]



# fig, ax = plt.subplots()
width = 7.20472
fig = plt.figure(figsize=(width, 0.6*width))
gs = fig.add_gridspec(3, 2, hspace=0.00, wspace=0.3, width_ratios=[2.1, 1])

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[2,0])

colors = ['C0', 'C1']

axes=[ax1, ax2, ax3]

datafile = h5py.File("FigureData.h5", 'a')
grp = datafile.create_group("Figure 3a")

grp.attrs["Explanation"] = "Column 1 = forcing frequency (x-axis), column 2 = tidal amplitude (equilibrium), column 3 = tidal amplitude (dynamic)"

labels = ['80 km', '50 km']
lines = []
for x in range(2):
    infile = h5py.File(file_names[x], 'r')
    # infile2 = h5py.File(file_names[x], 'r')
    # infile = h5py.File("/home/hamish/Research/LaplaceTidalAnalytic/workflow/testin.h5", 'r')

    ho = infile['europa']['ocean thickness'][:]
    freqs = infile['europa']['frequency'][:]

    

    idx = (np.abs(ho - target_h[x])).argmin()
    # idx -= 1
    print(freqs)
    print(ho[idx])

    phi = infile['europa']['velocity potential'][idx]

    infile.close()

    n_max = len(phi[0, 0, :])
    n = np.arange(1, n_max+1, 1)

    ff, nn = np.meshgrid(freqs, n)

    Nnm0 = n*(n+1)/(2*n+1)*factorial(n)/factorial(n)
    Nnm2 = n*(n+1)/(2*n+1)*factorial(n+2)/factorial(n-2)
    Nnm2[0] = 1.0

    eta_freq = np.zeros(len(freqs))
    eta_freq_eq = np.zeros(len(freqs))
    eta_max = 0

    for i in range(len(eta_freq)):
        cilm = np.zeros((2, n_max+1, n_max+1))
        eta_eq_cilm = np.zeros((2, n_max+1, n_max+1))

        # phi_m2 = np.nansum( phi[i,2,:] * n*(n+1)/r**2.0 * (ho[idx]/(-(force_freq)*freqs[i])), axis=0)
        # phi_m0 = np.nansum( phi[i,0,:] * n*(n+1)/r**2.0 * (ho[idx]/(-(force_freq)*freqs[i])), axis=0)
        phi_m2 = phi[i,2,:] * n*(n+1)/r**2.0 * (ho[idx]/(-(force_freq)*freqs[i]))
        phi_m0 = phi[i,0,:] * n*(n+1)/r**2.0 * (ho[idx]/(-(force_freq)*freqs[i]))

        cilm[0,1:,2] = -phi_m2.imag
        cilm[1,1:,2] =  phi_m2.real
        cilm[0,1:,0] = -phi_m0.imag
        cilm[1,1:,0] =  phi_m0.real


        eta_eq_cilm[0, 2, 2] = U22_C[i] + U22_S[i]
        eta_eq_cilm[0, 2, 0] = U20[i]

        # print(cilm)

        griddh = MakeGridDH(cilm, norm=3, sampling=2)
        if i == 0:
            displacement = griddh.copy()
            disp_coeff = cilm.copy()

        else:
            displacement += griddh
            disp_coeff += cilm

        eta_freq[i] = np.amax(np.amax(abs(griddh)))

        # print(np.shape(griddh))
        val = 6
        eta_freq[i] = abs(griddh[val,0])
        # print(freqs[i], abs(griddh[val,0]))
        # eta_max += griddh[4,4]

        griddh_eq = MakeGridDH(eta_eq_cilm, norm=3, sampling=2)

        eta_freq_eq[i] = abs(griddh_eq[val,0]) #13

    # fig, ax = plt.subplots()
    # ax.contourf(displacement)
    # # fig, ax= plt.subplots()
    # fig.savefig("/home/hamish/Dropbox/Tests/disp_test.pdf")

    d = np.vstack((freqs.T/2, eta_freq_eq.T, eta_freq.T)).T
    grp.create_dataset("tidal amplitude for h=" + str(int(target_h[x]/1e3)) +" km", data=d)

    l_eq, = axes[x].step(freqs/2+0.25, eta_freq_eq[::-1], 'k-', lw=0.4)

    l_h = axes[x].fill_between(freqs/2+0.25, eta_freq[::-1], facecolor=colors[x], step="pre",  alpha=0.8)

    ax3.fill_between(freqs/2+0.25, eta_freq[::-1]/eta_freq_eq[::-1], facecolor=colors[x], step="pre", alpha=0.8)



    if x is 0:
        lines.append(l_eq)

    lines.append(l_h)


axes[0].legend(lines, ['Equilibrium tide', 'Dynamic tide, $h_o=$ \\SI{80}{\\km}', 'Dynamic tide, $h_o=$ \\SI{50}{\\km}'], frameon=False, prop={'size': 6}, loc='upper right')

for ax in axes:
    ax.set_yscale('log')
    ax.set_ylim([1e-6, 35])
    ax.set_xlim([-25, 25])
    ax.tick_params(axis='both', which='major', labelsize=ticksize)

    # ax.xaxis.set_tick_params(which='minor', top = True, width=0.5, direction='out', zorder=100000)

for ax in axes[:2]:
    ax.axes.xaxis.set_ticklabels([])
    ax.set_ylabel("Tidal\namplitude [m]", fontsize=labelsize)

    locmin = LogLocator(base=10.0, subs=(1.0, ), numticks=100)
    ax.yaxis.set_major_locator(locmin)
    labels = ["\\num[retain-unity-mantissa = false]{{{:1.0e}}}".format(item) for item in ax.get_yticks().tolist()]
    labels[::2] = [''] * len(labels[::2])
    ax.set_yticklabels(labels, fontsize=ticksize)

    # ax.yaxis.set_minor_locator(LogLocator(base=10,numticks=12))
    # ax.yaxis.set_tick_params(which='minor', direction='out', width=0.5, length=1.5)
    # ax.xaxis.set_tick_params(which='minor', bottom = False)
    # ax.axes.yaxis.set_ticklabels([])

ax3.set_ylim([1.0, 300])
ax3.set_xlabel("Tidal frequency, $q$", fontsize=labelsize)
ax3.set_ylabel("Dynamic tide\nperturbation, $\\eta/\\eta_{eq}$", fontsize=labelsize)

ax3.xaxis.set_minor_locator(MultipleLocator(1))
ax3.xaxis.set_tick_params(which='minor', direction='out', top=False, width=0.5, length=1.5)
# locmin = LogLocator(base=10,numticks=12)
# ax3.yaxis.set_minor_locator(locmin)
ax3.yaxis.set_tick_params(which='minor', direction='out', width=0.5, length=1.5)

labels = ["{:d}".format(int(item)) for item in ax3.get_xticks().tolist()]
ax3.set_xticklabels(labels, fontsize=ticksize)
labels = ["{:d}".format(int(item)) for item in ax3.get_yticks().tolist()]
ax3.set_yticklabels(labels, fontsize=ticksize)


ax4 = fig.add_subplot(gs[:,1])

N = 2
data = np.loadtxt("europa_res_68km.txt").T
lons = np.arange(0, 360., 360/np.shape(data)[1], dtype=float)
colats = np.arange(0.0, 180., 180/np.shape(data)[0], dtype=float)
# fig, ax = plt.subplots(figsize=(4,8))

# data = np.loadtxt("europa_res_68km.txt").T
data_io = np.loadtxt("io_res_46km.txt").T

data_merid = np.mean(data, axis=1)
data_merid_io = np.mean(data_io, axis=1)

print(np.mean(data_merid),np.mean(data_merid_io))

data_mean = np.mean(data_merid)
data_mean_io = np.mean(data_merid_io)



p = ax4.plot(data_merid/data_mean, 90-colats, lw=1.8, label = 'Europa, $h_0 =$ \\SI{68.103}{km}')
p = ax4.plot(data_merid_io/data_mean_io, 90-colats, lw=1.8, label = 'Io, $h_0 =$ \\SI{46.489}{km}')
ax4.set_ylim([-90,90])

grp = datafile.create_group("Figure 3b")
grp.attrs["Explanation"] = "Column 1 = latitudes (y-axis), column 2 = europa, column 3 = io"

d = np.vstack((90-colats.T, data_merid.T, data_merid_io.T)).T
grp.create_dataset("Oceanic tidal heat flux", data=d)

ax4.xaxis.set_major_locator(MultipleLocator(0.25))
ax4.xaxis.set_tick_params(which='minor', direction='out', top=False, width=0.5, length=1.5)

labels = ["{:.2f}".format(item) for item in ax4.get_xticks().tolist()]
ax4.set_xticklabels(labels, fontsize=ticksize)
# labels = ["{:d}".format(int(item)) for item in ax4.get_yticks().tolist()]
# ax4.set_yticklabels(labels, fontsize=ticksize)

ax4.set_ylabel("Latitude [\\si{\\degree}]", fontsize=labelsize)
ax4.set_xlabel("Normalized zonally-averaged heat flux", fontsize=labelsize)

ax4.legend(frameon=False, prop={'size': 6}, loc='upper right')

# font = font_manager.FontProperties(family='STIXGeneral',
#                                    weight='bold',
#                                    style='normal', size=12)

tt = ax1.text(-0.12, 1.0, "\\textbf{a}", fontfamily="sans-serif", transform=ax1.transAxes,
            size=11, weight='bold')

tt = ax4.text(1.1, 1.0, "\\textbf{b}", fontfamily="sans-serif", transform=ax1.transAxes,
            size=11, weight='bold')


# plt.setp(tt.texts, family='Consolas')

fig.savefig("/home/hamish/Dropbox/Tests/disp_europa.pdf", bbox_inches='tight', dpi=1000)

# disp_coeff = pyshtools.expand.SHExpandDH(displacement, norm=3, sampling=2)
#
# fig, ax = plt.subplots()
#
# power_per_l = pyshtools.spectralanalysis.spectrum(disp_coeff, normalization='unnorm')
# degrees = np.arange(disp_coeff.shape[1])
#
# # cnt = ax.semilogy(degrees[:8], power_per_l[:8], 'o')
# # ax.set_ylim([1e-8, 1e5])
#
# cnt = ax.pcolormesh(displacement)
# plt.colorbar(cnt)
#
# fig.savefig("/home/hamish/Dropbox/Tests/disp_europa2.pdf", bbox_inches='tight')

plt.show()
