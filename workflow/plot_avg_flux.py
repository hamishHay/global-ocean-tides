import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import planetPy
from scipy.interpolate import interp1d

save = False

def get_data(file_name, moon):
    if isinstance(file_name, list):
        if len(file_name) == 1:
            file_name = file_name[0]
    
            data_file = h5py.File(file_name, 'r')

            ho = data_file[moon]['ocean thickness'][:]
            diss_ocean = np.sum(data_file[moon]['dissipated power: ocean'][:,:], axis=0)
            diss_shell = np.sum(data_file[moon]['dissipated power: shell'][:,:], axis=0)

            data_file.close()
        else:
            hos = []
            diss_os = []
            diss_sh = []
            interp_os = []
            interp_sh = []

            for i in range(len(file_name)):
                print(file_name[i])
                data_file = h5py.File(file_name[i], 'r')

                hos.append(data_file[moon]['ocean thickness'][:])
                diss_os.append(np.sum(data_file[moon]['dissipated power: ocean'][:,:], axis=0))
                diss_sh.append(np.sum(data_file[moon]['dissipated power: shell'][:,:], axis=0))

                interp_os.append(interp1d(hos[-1], np.log10(diss_os[-1])))
                interp_sh.append(interp1d(hos[-1], np.log10(diss_sh[-1])))

                data_file.close()

            ho = np.unique(np.concatenate(hos))
            diss_ocean = np.zeros(len(ho))
            diss_shell = np.zeros(len(ho))

            for i in range(len(file_name)):
                diss_ocean += 10**interp_os[i](ho)
                diss_shell += 10**interp_sh[i](ho)

    else:
        data_file = h5py.File(file_name, 'r')

        ho = data_file[moon]['ocean thickness'][:]
        diss_ocean = np.sum(data_file[moon]['dissipated power: ocean'][:,:], axis=0)
        diss_shell = np.sum(data_file[moon]['dissipated power: shell'][:,:], axis=0)

        data_file.close()

    return ho/1e3, diss_ocean, diss_shell

# plt.style.use("dark_background")

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx}, \\usepackage{libertine}, \\usepackage{libertinust1math}, \\sisetup{detect-all}')

plt.rc('lines', linewidth=0.6)

labelsize = 9
ticksize = 8

moons = ['io', 'europa', 'ganymede', 'callisto']
eq_flux = [0.1122, 0.05745713826366558, 0.005709, 0.005709]
eq_flux = [0.10912028985507247, 0.05658004244758728, 0.0054779721352382835, 0.005458137785261199]
rads = [[7.46395e13, 1.66792e14], [200e9,200e9*1.5], [400e9,400e9*1.5], [400e9,400e9*1.5]]

# ecc_files = ['io_ecc_v1e20_a1e-11.h5', 'europa_ecc_v1e18_a1e-11.h5', 'ganymede_ecc_v1e18_a1e-11.h5', 'callisto_ecc_v1e18_a1e-11.h5']

# moon_files = [['io_v1e20_a1e-11.h5'], 
#               ['europa_v1e18_a1e-11.h5'], 
#               ['ganymede_v1e18_a1e-11.h5', 'ganymede_v1e18_a1e-11_callisto.h5'],
#               ['callisto_v1e18_a1e-11.h5', 'callisto_ecc_v1e18_a1e-11.h5']]

ecc_files = ['io_fecc_v1e21_a1e-11_maxwell.h5', 'europa_fjup_v1e17_a1e-11_maxwell.h5', 'gan_fjup_v1e17_a1e-11_maxwell.h5', 'callisto_fjup_v1e17_a1e-11_maxwell.h5']

moon_files = [['io_feur_fecc_v1e21_a1e-11_maxwell.h5'], 
              ['europa_fio_fjup_fgan_v1e17_a1e-11_maxwell.h5'], 
              ['gan_feur_fjup_v1e17_a1e-11_maxwell.h5', 'gan_fcal_v1e17_a1e-11_maxwell.h5'],
              ['callisto_fgan_v1e17_a1e-11_maxwell.h5', 'callisto_fjup_v1e17_a1e-11_maxwell.h5']]

info_dict = {'io':{}, 'europa':{}, 'ganymede':{}, 'callisto':{}}
for i in range(len(moons)):
    info_dict[moons[i]]['eq flux'] = eq_flux[i]
    info_dict[moons[i]]['radiogenic'] = rads[i]
    info_dict[moons[i]]['ecc file'] = ecc_files[i]
    info_dict[moons[i]]['moon file'] = moon_files[i]


data = planetPy.database.Database()
moon_dict = data.jupiter.moons

cmap = plt.cm.get_cmap('plasma')
c1 = cmap(0.0)
c2 = cmap(0.75)#0.75

# c1 = cmap(0.5)

width = 3.50394
fig, axes = plt.subplots(nrows=4, figsize=(width,width*2.5))


if save:
    datafile = h5py.File("FigureData.5", 'a')
    grp = datafile.create_group("Figure 2")

    grp.attrs["Explanation"] = "Column 1 = ocean thickness (x-axis), column 2 = dissipated power"


moons=['io', 'europa','ganymede', 'callisto']
for i in range(len(moons)):
# for i in range(len(moons)):
    moon = moons[i]

    R = moon_dict[moon.capitalize()].radius
    area = 4*np.pi*R**2.0

    ho, d_ocean, d_shell = get_data(info_dict[moon]['moon file'], moon)
    diss_data = d_ocean + d_shell

    l1, = axes[i].loglog(ho, diss_data, color=c2, lw=1.1)#, alpha=0.0)

    if save:
        d = np.vstack((ho.T, diss_data.T)).T
        grp.create_dataset(moon + " heating with " + "jupiter and moon-moon forcing", data=d)

    ho, d_ocean, d_shell = get_data(info_dict[moon]['ecc file'], moon)
    diss_data = d_ocean + d_shell
    # diss_data = d_shell

    l2, = axes[i].loglog(ho, diss_data, color=c1, lw=1.1)

    if save:
        d = np.vstack((ho.T, diss_data.T)).T
        grp.create_dataset(moon + " heating with " + "jupiter forcing", data=d)

    axes[i].set_xlim([ho.min(), ho.max()])
    axes[i].set_ylim(top=1e16, bottom=1e6)


    ax = axes[i].twinx()
    ax.set_yscale('log')
    ax.set_ylim( np.array(axes[i].get_ylim())/area )

    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    axes[i].yaxis.set_major_locator(locmaj)

    # locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.1,)) 
    # axes[i].yaxis.set_minor_locator(locmin)

    ticksize = 9
    labels = ["\\num[retain-unity-mantissa = false]{{{:1.0e}}}".format(item) for item in axes[i].get_yticks().tolist()]
    axes[i].set_yticklabels(labels, fontsize=ticksize)

    labels = ["\\num[retain-unity-mantissa = false]{{{:1.0e}}}".format(item) for item in ax.get_yticks().tolist()]
    ax.set_yticklabels(labels, fontsize=ticksize)



    axes[i].set_ylabel("Tidal power [W]", fontsize=labelsize)
    ax.set_ylabel("Tidal heat flux [W m$^{-2}$]", fontsize=labelsize)

    axes[i].text(0.98, 0.88, moon.capitalize(), transform = ax.transAxes, ha='right', fontsize=ticksize)

    ax.axhline(info_dict[moon]['eq flux'], linestyle='--', color='k', linewidth=1.2)

    if i> 0:
        axes[i].axhline(info_dict[moon]['radiogenic'][0],color=(0.1, 0.1, 0.1, 0.6), linewidth=1.2)
    else:
        axes[i].fill_between(ho, info_dict[moon]['radiogenic'][0], info_dict[moon]['radiogenic'][1], color=(0.1, 0.1, 0.1, 0.6), linewidth=0.0)

    axes[i].set_xticklabels([0.1, 0.1, 1, 10, 100], fontsize=ticksize)

    if i is 0:
        axes[i].legend([l2, l1], ['Jupiter forcing only', 'Jupiter+moon forcing'], frameon=False, prop={'size': 8}, loc=4)

# ax2.set_ylim([1e6, 1e16])
for ax in axes:
    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
#     ax.set_xlim([0.1, 150])
#     ax.set_ylabel("Tidal power [W]", fontsize=labelsize)

axes[-1].set_xlabel("Ocean thickness [km]", fontsize=labelsize)



# plt.subplots_adjust(hspace=0.15)

fig.savefig("/home/hamish/Dropbox/Tests/new_diss_plot_rev3.pdf", dpi=400, bbox_inches='tight')#, transparent=True)
plt.show()
