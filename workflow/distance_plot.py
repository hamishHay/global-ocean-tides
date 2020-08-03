from LTE_solver import LTESolver
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from scipy.misc import factorial
from pyshtools.expand import MakeGridDH
import sys
import planetPy
import loveNumbers
from planetPy.constants import G

def create_dataset(grp, tag, data, type):
    """ Save an array (data) to an hdf5 group (grp) with the name tag (tag) abs
    with type (type). If the group already exists then the existing array is
    overwritten. If the data array mismatches what already exists, then the
    data is also overwritten.
    """

    try:
        grp.create_dataset(tag, data=data, dtype=type)
    except RuntimeError:
        if np.shape(data) is not np.shape(grp[tag]):
            del grp[tag]
            grp.create_dataset(tag, data=data, dtype=type)
        else:
            grp[tag][...] = data

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

plt.rc('axes', linewidth=1.5)

plt.rc('figure', dpi=400)
plt.rcParams['pdf.fonttype'] = 42



time = np.linspace(0, 1.0, 10000)

europa = planetPy.database.Database().europa 
io = planetPy.database.Database().io

e = europa.eccentricity
a = europa.semimajor_axis 
n = europa.mean_motion
P = 2*np.pi/n
Re = europa.radius

a_io = io.semimajor_axis

R = io.parent.radius
Mj = io.parent.mass
Mio = io.mass

time = np.linspace(-1.5, 1.5, 10000)*P

M = np.linspace(0, 2*np.pi, 10000)


r_ecc = a/(1+e*np.cos(M))

r_io = np.sqrt(a**2.0 + a_io**2.0 - 2*a*a_io*np.cos(n*time))

plt.plot(time/P, (R/r_ecc)**3.0)
plt.plot(time/P, (R/r_io)**3.0)

ax = plt.gca() 

ax.set_xlim([-1.5, 1.5])
ax.set_xlabel("Orbital phase")
ax.set_ylabel("$1/r^3$ [$R_{jup}^{-3}$]")

ax.set_title("Inverse Distance Cubed at Europa")

fig = plt.gcf() 

fig.savefig('/home/hamish/Dropbox/Tests/dist.pdf', bbox_inches='tight')