import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py

matplotlib.rcParams.update({'font.size': 14})

plt.style.use('dark_background')

infile = h5py.File('plot_diss.h5', 'r')

fig, ax = plt.subplots(figsize=(10,3.5))

diss = infile['europa']['eccentricity']['dissipated power'][:,:]
diss_ecc = diss[98, :]
# diss_io = diss[120, :]

ho = infile['europa']['eccentricity']['ocean thickness'][:]/1e3

ax.loglog(ho, diss_ecc, c='C1')
# ax.loglog(ho, diss_io, c='C0')

infile = h5py.File('plot_diss_a6.h5', 'r')

diss = infile['europa']['eccentricity']['dissipated power'][:,:]
diss_ecc = diss[98, :]
# diss_io = diss[120, :]

ho = infile['europa']['eccentricity']['ocean thickness'][:]/1e3

ax.loglog(ho, diss_ecc, ':', dashes=(1.2, 1.2))#, lw=0.8)#, c='C1')
# ax.loglog(ho, diss_io, 'k:', dashes=(0.8, 0.8), lw=0.8)#, c='C0')


ax.set_xlim([0.1, 150])
ax.set_ylim([1e6, 5e18])
#
#
# ax2.set_ylim([-fmax,fmax])
# ax2.yaxis.set_ticklabels([])
# ax2.legend(frameon=False, prop={'size': 8})
#
# ax2.grid(which='major', alpha=0.4)
# #
# #
# ax1.set_ylabel("Forcing frequency, $q$")
# ax1.tick_params(labelbottom=False)
ax.set_xlabel("Ocean thickness [km]")
#
# # ax2.yaxis.tick_right()
# #
# ax2.set_xscale("log")
# ax2.set_xlim([1e-5, 100])
# ax2.set_ylim([qmin,qmax])
# ax2.set_xlabel("Tidal potential [m$^2$ s$^{-2}$]")
#
# locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
# ax2.xaxis.set_minor_locator(locmaj)
# ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# ax4.grid(which='both', alpha=0.4)
#
ax.set_ylabel("Tidal power [W]")
#
#
# ax4.text(0.2, 4e2, '$q=10$, $\\alpha=10^{-9}$', fontsize=7, rotation=4)
# ax4.text(0.2, 4e5, '$q=10$, $\\alpha=10^{-6}$', fontsize=7, rotation=3.5)
# ax4.text(0.75, 6e9, '$q=1$, $\\alpha=10^{-9}$', fontsize=7, rotation=-4.5)
# ax4.text(0.75, 1e13, '$q=1$, $\\alpha=10^{-6}$', fontsize=7, rotation=-4.5)
#
# fig.suptitle("Europan tides and response, 20 km shell", y=0.915)
#
# fig = plt.gcf()
fig.savefig("/home/hamish/Dropbox/Tests/lplc_ecc.png", dpi=600, bbox_inches='tight', transparent=True)

plt.show()
