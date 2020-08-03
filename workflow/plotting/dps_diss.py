import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

alphas = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10][::-1]

c = np.arange(1, len(alphas) + 1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
cmap.set_array([])


fig, ax = plt.subplots(figsize=(4,4), dpi=100)
for i in range(len(alphas)):
    infile = h5py.File("DPS_"+str(alphas[i])+".h5", 'r')

    data = infile['europa']['io']['dissipated power']
    ho = infile['europa']['io']['ocean thickness'][:]/1e3

    ax.semilogy(ho, data, c=cmap.to_rgba(i + 1))

ax.set_xlim([90, 120])

ax.set_xlabel("Ocean thickness [km]")
ax.set_ylabel("Dissipated Power [W]")


cb = fig.colorbar(cmap, ticks=c, orientation='horizontal', pad=0.2, shrink=0.8)
cb.set_ticklabels(["$10^{-10}$", "$10^{-9}$", "$10^{-8}$", "$10^{-7}$", "$10^{-6}$", "$10^{-5}$"])
cb.set_label("Drag coefficient [s$^{-1}$]")

fig.savefig("/home/hamish/Dropbox/Tests/dps_plot.pdf", bbox_inches='tight')
