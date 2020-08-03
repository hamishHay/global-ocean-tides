import  numpy as np 
import matplotlib.pyplot as plt 
import h5py 
from planetPy.database import Database

fig, axes = plt.subplots(nrows=2,sharex=True)
ax1, ax2 = axes

data = Database()

moon = data.europa
R = data.europa.parent.radius
files = ["europa_io_vs_sma_1e-11_1e19_ho30km.h5", "europa_io_vs_sma_1e-11_1e19_ho28.25km.h5"]
i = 0
for ax in axes:
    infile = h5py.File(files[i])
    a = infile['europa']['semimajor axis'][:]
    diss_o = np.sum(infile['europa']['dissipated power: ocean'][:], axis=0)
    diss_c = np.sum(infile['europa']['dissipated power: shell'][:], axis=0)
    infile.close()



    ax.semilogy(a/R, diss_o, label='ocean')
    ax.semilogy(a/R, diss_c, '--', label='crust')

    ax.axvline(moon.semimajor_axis/R, linestyle='--', color='k', linewidth=0.8)
    i += 1

    ax.set_ylim([1e2,1e10])
    ax.set_xlim([(a/R).min(), (a/R).max()])

ax2.legend(frameon=False, loc='upper right')

ax2.set_xlabel("Europa's semimajor axis [$a/R_{j}$]")
ax2.set_ylabel("Tidal power [W]")
ax1.set_ylabel("Tidal power [W]")

ax1.text(0.01,0.04,'$h_o =$ 30.00 km',transform=ax1.transAxes)
ax2.text(0.01,0.04,'$h_o =$ 28.25 km',transform=ax2.transAxes)

# labels = [item.get_text() for item in ax2.get_yticklabels()]
labels=ax2.get_yticks().tolist()
# labels[1]=''

labels = ["$10^{{ {:d} }} $".format(int(np.log10(int(i)))) for i in labels]

labels[-1] = ''
ax2.set_yticklabels(labels)

plt.subplots_adjust(hspace=0.1)

fig.savefig("/home/hamish/Dropbox/Tests/diss_vs_a.pdf", bbox_inches='tight')