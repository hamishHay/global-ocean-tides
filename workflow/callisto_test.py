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

generating_text = " ".join(sys.argv[:])
print(generating_text)

import argparse

parser = argparse.ArgumentParser(description='Calculate LTE.')
parser.add_argument('body_name', metavar='B', type=str,
                    help='tidal deformed body')
# parser.add_argument('forcing_type', metavar='F', type=list,
                    # help='type of forcing')
parser.add_argument('-f','--forcing_name', action='append', help='<Required> Set flag', required=True, type=str)
parser.add_argument('--drag', type=float,
                    help='drag coefficient',default=1e-6)
parser.add_argument('--shell', type=float, default=1.0,
                    help='ice shell thickness')
parser.add_argument('--density', type=float, default=1e3,
                    help='fluid density')
parser.add_argument('--hmin', type=float,
                    help='minimum ocean thickness',default=1e2)
parser.add_argument('--hmax', type=float,
                    help='maximum ocean thickness',default=1e3)
parser.add_argument('--hnum', type=int,
                    help='ocean thickness number',default=101)
parser.add_argument('--qmax', type=int,
                    help='maximum frequency', default=15)
parser.add_argument('--qmin', type=int,
                    help='minimum frequency', default=1)
parser.add_argument('--nmax', type=int,
                    help='maximum spherical harmonic degree', default=12)
parser.add_argument('--save', type=bool, default=False,
                    help='savefile name')
parser.add_argument('--save_name', type=str, default=None,
                    help='savefile name')

args = parser.parse_args()

n_max = args.nmax
alpha = args.drag
q_max = args.qmax
q_min = args.qmin
h_min = args.hmin
h_max = args.hmax
h_num = args.hnum
h_shell = args.shell
save  = args.save
save_name = args.save_name
den = args.density


data = planetPy.database.Database()
moons = data.moon_dict
# moons = data.planet_dict

moon_name = args.body_name.lower()
forcing_name = args.forcing_name#.lower()
print(forcing_name)

# print(forcing_name + " forcing on " + moon_name )
# sys.exit()

if save_name is None:
    save_name = "LTE_solutions_alpha_" + str(alpha) + ".h5"

# print(save_name)
rot_rate = moons[moon_name.capitalize()].mean_motion
# rot_rate = 1e-8

# if forcing_name == 'eccentricity' or forcing_name == 'obliquity':
#     conj_freq = rot_rate
# else:
#     conj_freq = abs(rot_rate - moons[forcing_name.capitalize()].mean_motion)


# if moon_name is 'europa' or moon_name is 'io':
#     conj_freq = abs(rot_rate - moons['Ganymede'].mean_motion)




radius = moons[moon_name.capitalize()].radius - h_shell*1e3
grav =  moons[moon_name.capitalize()].gravity
den_bulk = moons[moon_name.capitalize()].density
den_ocean = 1000.0

print(rot_rate)

if save:
    save_file = h5py.File(save_name,'a')

    try:
        grp = save_file.create_group(moon_name)
    except ValueError:
        grp = save_file[moon_name]

    # try:
    #     sgrp = grp.create_group(forcing_name)
    # except ValueError:
    #     sgrp = grp[forcing_name]


ho = np.logspace(np.log10(h_min), np.log10(h_max), h_num)

ho = np.arange(1, 300.1, 1.0)*1e3

h_shell = int(h_shell)

# beta = np.loadtxt("/home/hamish/Research/Io/ocean_dissipation_ivo/beta_"+moon_name+".txt")[:, h_shell-1]
# ups = np.loadtxt("/home/hamish/Research/Io/ocean_dissipation_ivo/upsilon_"+moon_name+".txt")[h_shell-1]

# love_file = h5py.File("shell_love_numbers.h5", 'r')
# beta = love_file[moon_name]["beta"][h_shell - 1]
# ups = love_file[moon_name]["upsilon"][h_shell - 1][1]
#
# # beta =
# beta[0] = 0.00001#0.0001
# beta[:] = 1.0

# kL, hL, gamma_L = loveNumbers.loading_surface_ocean(den_bulk, grav, radius, 40e9, nmax=n_max+6)
# kT, hT, gamma_T = loveNumbers.tidal_surface_ocean(den_bulk, grav, radius, 40e9)

# love_file = h5py.File("shell_love_numbers.h5", 'r')
# beta = love_file[moon_name]["beta"][h_shell - 1][:]
# ups = love_file[moon_name]["upsilon"][h_shell - 1][1]
# beta[0] = 0.000001
# beta[:] = 1.0

# gamma_T = (1+kT-hT)
# gamma_L = 1 - (1+kL-hL)

# print(gamma_L, gamma_T)
# forcings = ['io', 'ganymede']

coeff_file = h5py.File("galilean_coefficients.h5", 'r')
freqs  = coeff_file[moon_name][forcing_name[0]]['frequency q'][100-q_max:100+1-q_min]
freqs = np.concatenate((freqs, coeff_file[moon_name][forcing_name[0]]['frequency q'][100+q_min:100+1+q_max]))

q_nij  = coeff_file[moon_name][forcing_name[0]]['frequency'][100-q_max:100+1-q_min]
q_nij = np.concatenate((q_nij, coeff_file[moon_name][forcing_name[0]]['frequency'][100+q_min:100+1+q_max]))

coeffs1 = coeff_file[moon_name][forcing_name[0]]['tide coeffs'][100-q_max:100+1-q_min, :]
coeffs2 = coeff_file[moon_name][forcing_name[0]]['tide coeffs'][100+q_min:100+1+q_max, :]
if len(forcing_name) is not 0:
    for i in forcing_name[1:]:
        coeffs1 += coeff_file[moon_name][i]['tide coeffs'][100-q_max:100+1-q_min, :]
        coeffs2 += coeff_file[moon_name][i]['tide coeffs'][100+q_min:100+1+q_max, :]

coeffs = np.concatenate((coeffs1, coeffs2))
coeffs *= 0.5 #*ups
#
# print(freqs, coeffs[:,4])
# sys.exit()
res_list = []
forcing_magnitude = 1.0
# for h in ho:
# den_core = (den_bulk + den_ocean*((1 - ho/radius)**3.0 - 1))/(1 - ho/radius)**3.0
# #
# #     kL, hL, gamma_L = loveNumbers.loading_surface_ocean(den_core, grav, radius, 40e9, nmax=n_max+10)
# #     kT, hT, gamma_T = loveNumbers.tidal_surface_ocean(den_core, grav, radius, 40e9)
# #
# mass_core = 4./3.*np.pi*(radius-ho)**3.0*den_core
# mass_ocean = 4./3.*np.pi*(radius**3.0 - (radius-ho)**3.0)*den_ocean

# print(ho[-1], mass_ocean[-1]/mass_core[-1])

solver = LTESolver(rot_rate, radius, grav, 1e3, alpha=alpha, nmax=n_max+5)
# solver.set_beta(beta)

print(grav)
for q in q_nij:
    # if abs(q) > 1e-10:
        # forcing_freq = (conj_freq)*q
    if q > 0.0:
        solver.define_forcing(forcing_magnitude, q, 2, 0)
        solver.setup_solver()
        resH = solver.find_resonant_thicknesses()

        res_list.append(resH)

    # solver.define_forcing(forcing_magnitude, -forcing_freq, 2, 2)
    # solver.setup_solver()
    # resH = solver.find_resonant_thicknesses()
    # res_list.append(resH)

    # print(q)
    solver.define_forcing(forcing_magnitude, q, 2, 2)
    solver.setup_solver()
    resH = solver.find_resonant_thicknesses()
    res_list.append(resH)

    solver.define_forcing(forcing_magnitude, q, 2, 1)
    solver.setup_solver()
    resH = solver.find_resonant_thicknesses()
    res_list.append(resH)

        #
        # solver.define_forcing(forcing_magnitude, forcing_freq, 2, 1)
        # solver.setup_solver()
        # resH = solver.find_resonant_thicknesses()
        # res_list.append(resH)
        #     # print(resH[1]/1e3)

ho_res = np.array(res_list)
ho_res = ho_res[ho_res>0.0]
# ho_res = ho_res[ho_res<400]
ho_res = ho_res[abs(ho_res)>np.amin(abs(ho))]
ho_res = ho_res[abs(ho_res)<np.amax(abs(ho))]
#
# for i in range(len(np.unique(ho_res))):
#     print(np.unique(ho_res/1e3)[i])

print(ho_res)
# ho = np.concatenate((ho, ho_res))
# ho = np.unique(ho)

# sys.exit()

if save:
    create_dataset(grp, 'frequency', freqs, np.int)
    create_dataset(grp, 'ocean thickness', ho, np.float)
    grp.attrs['build command'] = generating_text

n = np.arange(1, n_max+1, 1)

Nnm0 = n*(n+1)/(2*n+1)*factorial(n)/factorial(n)
Nnm2 = n*(n+1)/(2*n+1)*factorial(n+2)/factorial(n-2)
Nnm2[0] = 1.0
Nnm1 = n*(n+1)/(2*n+1)*factorial(n+1)/factorial(n-1)
# Nnm1[0] = 1.0
# Nnm0[0] *= 1

E22 = np.zeros(( len(freqs), len(ho)))
E20 = np.zeros(( len(freqs), len(ho)))
E21 = np.zeros(( len(freqs), len(ho)))
# E22_an = np.zeros(len(ho))
# E20_an = np.zeros(len(ho))

PHI = np.zeros( ( len(ho), len(freqs), 3, len(n) ), dtype=np.complex )
PSI = np.zeros( ( len(ho), len(freqs), 3, len(n) ), dtype=np.complex )

betas = np.loadtxt("/home/hamish/Dropbox/LPL/Icy Satellites/BetaUps/call_beta_constant_h20.csv", delimiter=',')
ups = np.loadtxt("/home/hamish/Dropbox/LPL/Icy Satellites/BetaUps/call_ups_constant_h20.csv", delimiter=',')

# betas[:,:] = 1.0
# ups[:, :] = 1.0

print(betas[0])


for i in range(len(ho)):
    ocean_thickness = ho[i]

    den_core = (den_bulk + den_ocean*((1 - ocean_thickness/radius)**3.0 - 1))/(1 - ocean_thickness/radius)**3.0

    # kL, hL, gamma_L = loveNumbers.loading_surface_ocean(den_core, grav, radius, 40e9, nmax=n_max+10)
    # kT, hT, gamma_T = loveNumbers.tidal_surface_ocean(den_core, grav, radius, 40e9)

    lte_radius = radius - (600e3 - ocean_thickness)
    solver = LTESolver(rot_rate, lte_radius, grav, ocean_thickness, alpha=alpha, nmax=n_max)
    solver.set_beta(betas[i])

    print("Solving h="+str(ocean_thickness/1e3), end="\r", flush=True)

    cilm = np.zeros((2, n_max+1, n_max+1))

    for j in range(len(freqs)):

        a_20c = coeffs[j][0]*ups[i][1]
        a_22c = coeffs[j][4]*ups[i][1]
        a_21c = coeffs[j][2]*ups[i][1]
        b_21s = coeffs[j][9]*ups[i][1]
        b_22s = coeffs[j][11]*ups[i][1]

        forcing_freq = -q_nij[j]

        # print(forcing_freq, a_22c + b_22s)
        # forcing_freq = -conj_freq*q

        # --------------------------------------------------------------
        # Solve for libration component, P22
        if abs(a_22c + b_22s) > 1e-11:
            solver.define_forcing( a_22c + b_22s, forcing_freq, 2, 2)
            solver.setup_solver()
            psi, phi = solver.solve_lte()

            if save:
                PHI[i, j, 2, :] = phi[:]
                PSI[i, j, 2, :] = psi[:]

            delta1 = np.sum(Nnm2 * (abs(phi)**2.0 + abs(psi)**2.0))

            E22[j, i] += delta1

        # if q<0 and delta1 < 0.0001*E22[i]:
        #     break

        # --------------------------------------------------------------
        # Solve for libration component, P21
        if abs(a_21c + b_21s) > 1e-11:

            # print(forcing_freq, a_21c + b_21s)
            solver.define_forcing(a_21c+b_21s , forcing_freq, 2, 1)
            solver.setup_solver()
            psi, phi = solver.solve_lte()

            if save:
                PHI[i, j, 1, :] = phi[:]
                PSI[i, j, 1, :] = psi[:]

            E21[j, i] += np.sum(Nnm1 * (abs(phi)**2.0 + abs(psi)**2.0))

        # --------------------------------------------------------------
        # Solve for radial component, P20
        #
        # This component is symetrical about q=0, so rather than sum over
        # positive and negative q, we sum only over positive q while
        # doubling the forcing
        # print(freqs[j], 2*a_20c, a_22c, b_22s, ((7/8)*(2.05e-5)**2.0*radius**2.0*0.0094)/b_22s )
        # if forcing_freq>0:
        if abs(a_20c) > 1e-11:
           solver.define_forcing(a_20c, forcing_freq, 2, 0)
           solver.setup_solver()

           psi, phi = solver.solve_lte()

           if save:
               PHI[i, j, 0, :] = phi[:]
               PSI[i, j, 0, :] = psi[:]

           delta2 = np.sum(Nnm0 * (abs(phi)**2.0 + abs(psi)**2.0))
           E20[j, i] += delta2

           # if delta1 < 0.0001*E22[i] and delta2 < 0.0001*E20[i]:
           #     break


ho = abs(ho)
E22 *= den * ho *2 *alpha*np.pi
E20 *= den * ho * 2 *alpha*np.pi
E21 *= den * ho * 2 *alpha*np.pi

# E22_an *= den * ho *2 *alpha*np.pi
# E20_an *= den * ho * 2 *alpha*np.pi

# grp.create_dataset("stream function", data=PSI)
# grp.create_dataset("velocity potential", data=PHI)
if save:
    create_dataset(grp, 'stream function', PSI, np.complex)
    create_dataset(grp, 'velocity potential', PHI, np.complex)
    create_dataset(grp, 'dissipated power', E20 + E21 + E22, np.float)

# print((E22_an+E20_an)/(E22+E20))

area = 4*np.pi*radius**2.0
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.loglog(ho/1e3, (np.sum(E22+E20+E21, axis=0))/1e9, '-', lw=1.5)

print((np.sum(E22+E20+E21, axis=0))[0]/1e9)
# plt.loglog(ho/1e3, E22[1]/area, '-', lw=0.9)
F =  0.1122#0.05745713826366558
#
# plt.axhline(F*(4*np.pi*radius**2.0)/1e9)
#
# vals = np.abs((E20+E22+E21)/1e9 - F*(4*np.pi*radius**2.0)/1e9)
# # idx = (np.abs((E20+E22+E21)/1e9 - F*(4*np.pi*radius**2.0)/1e9)).argmin()
#
# idx = np.argsort(vals)[1]
#
# print(ho[idx]/1e3)
#
# plt.loglog(ho/1e3, np.sum(E22, axis=0)+np.sum(E20, axis=0), lw=0.6, alpha=0.7, label="$P_{22}$")
# # plt.loglog(ho/1e3, E20, lw=0.6, alpha=0.7, label="$P_{20}$")
# # plt.loglog(ho/1e3, E22_an+E20_an, 'C1', lw=1.2, label="Analytical")
# ax = plt.gca()
ax.set_xlabel("Ocean thickness [km]", fontsize=13)
# ax.set_ylabel("Tidal Heat Flux [W m$^{-2}$]")
ax.set_xlim([np.amin(ho)/1e3, np.amax(ho)/1e3])
# # ax.set_xlim([0.1, 0.3])
# data = np.loadtxt(moon_name+"_ocean_ecc_flux.dat").T
# hn = data[0]
# dat = data[1]
# #
# # ax.loglog(hn, dat*4*np.pi*radius**2.0/1e9)
#
openFile = h5py.File("dissipation_data.h5", 'r')
# # #
dataGroup = openFile['europa']
#
ho = dataGroup['ocean thickness'][:]
data = dataGroup['time avg dissipation'][0,:]

# print(data)
#
# ax.semilogy(ho/1e3, data, 'bo', markersize=2, label='ODIS')

ax.legend(frameon=False)
ax.set_ylabel("Tidal power [GW]", fontsize=13)
# ax.set_title(forcing_name.capitalize() + " forcing on " + moon_name.capitalize())
ax.grid(which='major', alpha=0.2)
# ax.get_xaxis().get_major_formatter().set_useOffset(False)
# fig = plt.gcf()
fig.savefig("/home/hamish/Dropbox/Tests/"+ save_name + ".pdf", dpi=200, bbox_inches='tight')
plt.show()
