import numpy as np
import matplotlib.pyplot as plt
import h5py
import planetPy
from scipy.interpolate import interp1d

def get_data(file_name):
    data_file = h5py.File(file_name, 'r')

    ho = data_file['europa']['ocean thickness'][:]
    diss_ocean = np.sum(data_file['europa']['dissipated power: ocean'][:,:], axis=0)
    diss_shell = np.sum(data_file['europa']['dissipated power: shell'][:,:], axis=0)

    data_file.close()

    return ho/1e3, diss_ocean, diss_shell

# plt.style.use("dark_background")

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx}, \\usepackage{libertine}, \\usepackage{libertinust1math}, \\sisetup{detect-all}')

plt.rc('lines', linewidth=0.6)

labelsize = 11
ticksize = 8

moons = ['io', 'europa', 'ganymede', 'callisto']
eq_flux = [0.1122, 0.05745713826366558, 0.005709, 0.005709]
rads = [[7.46395e13, 1.66792e14], [200e9,200e9*1.5], [400e9,400e9*1.5], [400e9,400e9*1.5]]

data = planetPy.database.Database()
moon_dict = data.jupiter.moons

cmap = plt.cm.get_cmap('plasma')
c1 = cmap(0.0)
c2 = cmap(0.9)#0.75

c1 = cmap(0.5)

width = 3.50394
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(width*2.0,width), sharex=True)

# data_file = h5py.File("dissipation_ecc_only.h5", 'r')
# data_file2 = h5py.File("dissipation_w_moons.h5", 'r')
# data_file3 = h5py.File("dissipation_w_moons_gan.h5", 'r')



ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]

axes = [ax1, ax2, ax3, ax4]

# ax2.fill_between(ho, rads[1][0], rads[1][1], color=(0.1, 0.1, 0.1, 0.6), linewidth=0.0)

# ax2.axhline(rads[1][0], color=(0.1, 0.1, 0.1, 0.6))

ho, d_ocean, d_shell = get_data("europa_v1e18_a1e-11.h5")
ax1.loglog(ho, d_shell, label='crust')
ax1.loglog(ho, d_ocean, label='ocean')
ax3.loglog(ho, d_shell/d_ocean)
ax1.legend(prop={'size': 6}, frameon=False)

ho, d_ocean, d_shell = get_data("europa_v1e19_a1e-11.h5")
ax2.loglog(ho, d_shell, label='crust', zorder=10)
ax2.loglog(ho, d_ocean, label='ocean')
ax4.loglog(ho, d_shell/d_ocean)



ax1.set_ylim([1e6, 1e16])
ax2.set_ylim([1e6, 1e16])
for ax in axes:
    ax.set_xlim([0.1, 150])

ax3.set_xlabel("Ocean Thickness [km]", fontsize=labelsize)
ax4.set_xlabel("Ocean Thickness [km]", fontsize=labelsize)
    

ax1.set_ylabel("Tidal power [W]", fontsize=labelsize)
# ax2.set_ylabel("Tidal power [W]", fontsize=labelsize)

ax3.set_ylabel("Shell/Ocean\nPower ratio", fontsize=labelsize)
# ax4.set_ylabel("Shell/Ocean Power ratio", fontsize=labelsize)


ax3.set_ylim([0.1, 200])
ax4.set_ylim([0.1, 200])

ax1.set_title("Shell viscosity: $10^{18}$ Pa s", fontsize=labelsize)
ax2.set_title("Shell viscosity: $10^{19}$ Pa s", fontsize=labelsize)

# axes[-1].set_xlabel("Ocean thickness [km]", fontsize=labelsize)

plt.tight_layout()

plt.subplots_adjust(wspace=0.15,hspace=0.08)

fig.savefig("/home/hamish/Dropbox/Tests/diss_ratio_plot.pdf", dpi=400, bbox_inches='tight')#, transparent=True)
plt.show()
