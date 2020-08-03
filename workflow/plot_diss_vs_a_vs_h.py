import  numpy as np 
import matplotlib.pyplot as plt 
import h5py 
from planetPy.database import Database

fig, ax = plt.subplots()


data = Database()

moon = data.europa
R = data.europa.parent.radius

#a: 201, h: 201
a = np.linspace(0.99,1.01,801)*moon.semimajor_axis/R
h = np.linspace(20, 30, 801)

diss = np.loadtxt("diss_contour_a_vs_h.txt").T

c = ax.pcolormesh(a, h, diss, rasterized=True)
# ax.contour(a, h, diss, levels=[np.log10(1e9)], colors='k')

ax.axvline(moon.semimajor_axis/R, linestyle='--', color='k', linewidth=0.8)

plt.colorbar(c, ax=ax, label='$log_{10}$ (Tidal power [W])')

# ax.set_ylim([1e2,1e10])
# ax.set_xlim([(a/R).min(), (a/R).max()])

# ax2.legend(frameon=False, loc='upper right')

# ax2.set_xlabel("Europa's semimajor axis [$a/R_{j}$]")
# ax2.set_ylabel("Tidal power [W]")
ax.set_ylabel("Ocean thickness [km]")
ax.set_xlabel("Europa's semimajor axis [$a/R_j$]")


# labels = [item.get_text() for item in ax2.get_yticklabels()]

# ax2.set_yticklabels(labels)

# plt.subplots_adjust(hspace=0.1)

fig.savefig("/home/hamish/Dropbox/Tests/diss_vs_a_h.pdf", bbox_inches='tight')
