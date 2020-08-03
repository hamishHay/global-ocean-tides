import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx}, \\usepackage{libertine}, \\sisetup{detect-all}')

plt.rc('lines', linewidth=0.6)

plt.set_cmap('bone')

N = 2
lons = np.arange(0, 360., N, dtype=float)
colats = np.arange(0.0, 180., N, dtype=float)
fig, ax = plt.subplots(figsize=(4,8))

data = np.loadtxt("europa_73km_res.txt")
data_io = np.loadtxt("io_80km_res.txt")


data_merid = np.mean(data, axis=1)
data_merid_io = np.mean(data_io, axis=1)

p = ax.plot(data_merid, 90-colats, lw=2.0, legend = 'Europa, $h_0 =$ \\SI{73.218}{km}')
p = ax.plot(data_merid_io, 90-colats, lw=2.0, legend = 'Io, $h_0 =$ \\SI{80.245}{km}')
ax.set_ylim([-90,90])

labels = ["{:.1f}".format(item) for item in ax.get_xticks().tolist()]
ax.set_xticklabels(labels)
labels = ["{:d}".format(int(item)) for item in ax.get_yticks().tolist()]
print(labels)
ax.set_yticklabels(labels)

ax.set_ylabel("Latitude [\\si{\\degree}]")
ax.set_xlabel("Meridional Average Heat Flux [\\si{\\watt\\per\\metre\\squared}]")


#
# cb = plt.colorbar(p, label='Heat Flux [\\si{\\watt\\per\\metre\\squared}]')
#
# # cb.ax.get_yticks()
# # print([item.get_text() for item in cb.ax.get_yticklabels()])
# labels = ["{:.2f}".format(item) for item in cb.ax.get_yticks().tolist()]
# cb.ax.set_yticklabels(labels)
#
fig.savefig("/home/hamish/Dropbox/Tests/diss_test.png", dpi=200, bbox_inches='tight')
