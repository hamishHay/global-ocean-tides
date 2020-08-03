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

#include "membraneConstants.h"
#include "array3d.h"
#include "array2d.h"
#include "array1d.h"
#include <iomanip>



def membraneUpsBeta(shell_thickness, ocean_thickness, freq, mass, radius, grav_surf, den_ocean, den_shell, n_max=12):
    def membraneFluidLoveNumbers(R, Rm, g, den_ocean, den_bulk, rigid):
        ''' Implementation of fluid love numbers - Eqs 57 and 58 from Beuthe (2015) - Tides on Europa'''

        # Define constants
        y = Rm/R 
        den_ratio = den_ocean/den_bulk 
        rigid_eff = rigid/(den_bulk * g * R)

        # Define polynomials from the propagator matrix (Eqs. 119, 120)

        A = 5./19. * (1 - den_ratio) * ( 2 + 3*y**5 - (2 - 5*y**3 + 3*y**5)*den_ratio )

        B = 10 - (16 - 25*y**3 + 9*y**5)*den_ratio + 3*(2 - 5*y**3 + 3*y**5)*den_ratio**2
        B *= 1./19. * (1 - den_ratio)

        h2_fluid = (A + 5*y**4*rigid_eff) / (B + (5 - 3*den_ratio) * y**4 * rigid_eff)
        k2_fluid = h2_fluid - 1.0 

        return k2_fluid, h2_fluid


    # d = np.loadtxt("love_numbers/europa_fluid_love_numbers.dat").T

    # k = d[1, :n_max]
    # h = d[2, :n_max]
    
    n = np.arange(1, n_max+1, 1)

    # sys.exit()


    G = 6.674e-11

    pois_ratio = 0.35
    rigid_shell = 3.5e9
    # rigid_shell = 60e9
    # rigid_shell  = (3.44 + 0.0001j) * 1e9

    visc_factor = rigid_shell/(freq*1e19)
    rigid_shell = (rigid_shell / ( 1 -  visc_factor*1j) )

    pois_ratio = (3*pois_ratio  - (1+pois_ratio)*visc_factor*1j) / (3 - 2*(1 + pois_ratio)*visc_factor*1j)

    # print(rigid_shell/1e9, pois_ratio)
    rigid_core = 40e9
    # pois_ratio = 0.5 
    # rigid_shell = 60e9

    # rigid_shell = (3.44 + 0.13j)*1e9
    # pois_ratio = 0.333 - 0.004j

    # rigid_shell = 3.5e9
    # pois_ratio = 0.33

    radius_core = radius #- ocean_thickness#(shell_thickness + ocean_thickness)
    radius_ocean = radius - (shell_thickness)

    beta = np.zeros(len(n), dtype=np.complex)
    ups = np.zeros(len(n), dtype=np.complex)


    mass_total = mass

    den_ocean = 1000.0
    den_shell = 1000.0

    vol_total = 4./3. * np.pi * radius**3.0


    den_bulk = mass_total/vol_total

    den_ratio= den_ocean/den_bulk


    spring = np.zeros(len(n), dtype=np.complex)
    for l in n:
        rigid_eff = (2*l*l + 4*l + 3)/l * rigid_core / (den_bulk * grav_surf * radius)

        rigid_factor = 1. / (1. + rigid_eff)

        # // FIND LOADING AND TIDAL LOVE NUMBERS
        
        if l==1:
            kt = 0.0 
            ht = 0.0
        else:
            ht = rigid_factor * (2*l + 1)/( ( 2 * (l-1) ) )
            kt = rigid_factor * 3./( ( 2 * (l-1) ) )

        kl = -rigid_factor
        hl = -rigid_factor * (2*l + 1) / 3.0

        gam_tide = 1.0 + kt - ht
        gam_load = 1.0 + kl - hl

        # // FIND THE SPRING CONSTANTS
        x = ((l - 1)*(l + 2))

        bendRigidity = rigid_shell * shell_thickness**3.0 \
                       / (6. * (1. - pois_ratio))

        sprMembrane = 2.*x * (1. + pois_ratio)/(x + 1. + pois_ratio) \
                      * rigid_shell / (den_ocean * grav_surf * radius) \
                      * shell_thickness / radius

        sprBending = x**2.0 * (x + 2.)/(x + 1. + pois_ratio) \
                     * bendRigidity / (den_ocean * grav_surf * radius**4.0)

        sprConst = sprMembrane + sprBending

        # // FIND dSpring and dGam^tide
        xi = 3.0 / (2.*l + 1.0) * (den_ratio) #// CHANGED den_core to den bulk and den_ocean to den_shell

        



        dsprConst = 1. - (1. + xi*hl)**2.0/(1. + xi*(ht - hl)*sprConst)
        dsprConst *= -sprConst

        dgam_tide = (1. + xi*hl)*ht / (1. + xi*(ht - hl)*sprConst)
        dgam_tide *= -sprConst

        # // Find beta and nu coefficients
        beta[l-1] = 1. - xi*gam_load + sprConst + dsprConst
        ups[l-1] = gam_tide + dgam_tide

        # Assume infinitely rigid mantle
        beta[l-1] = 1. - xi + sprConst 
        ups[l-1] = 1.0
        spring[l-1] = sprConst

        # print(spring)
    return beta, ups, spring



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

print(rot_rate)

# if forcing_name == 'eccentricity' or forcing_name == 'obliquity':
#     conj_freq = rot_rate
# else:
#     conj_freq = abs(rot_rate - moons[forcing_name.capitalize()].mean_motion)


# if moon_name is 'europa' or moon_name is 'io':
#     conj_freq = abs(rot_rate - moons['Ganymede'].mean_motion)




radius = moons[moon_name.capitalize()].radius# - h_shell*1e3
grav =  moons[moon_name.capitalize()].gravity
den_bulk = moons[moon_name.capitalize()].density
den_ocean = 1000.0
mass = moons[moon_name.capitalize()].mass
ecc = moons[moon_name.capitalize()].eccentricity

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

h_shell = float(h_shell)


beta, ups, spring = membraneUpsBeta(h_shell, 1e3, rot_rate, mass, radius, grav, 1e3, 1e3, n_max=n_max)

ups = ups[1]

beta[0] = 0.000001
print(beta[1])

coeff_file = h5py.File("galilean_coefficients.h5", 'r')
# coeff_file = h5py.File("enceladus_coefficients_real.h5", 'r')
# coeff_file = h5py.File("galilean_coefficients.h5", 'r')
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


# q_nij *= 1./3.

coeffs = np.concatenate((coeffs1, coeffs2))
coeffs = np.array(coeffs, dtype=np.complex)

# print(coeffs)
coeffs *= 0.5 *ups

coeff_file.close()

res_list = []
forcing_magnitude = 1.0


solver = LTESolver(rot_rate, radius, grav, 1e3, alpha=alpha, nmax=n_max)
solver.set_beta(beta)

SPRINGS = np.zeros( ( len(freqs),  4), dtype=np.complex)
for q in q_nij:

    beta, ups, spring = membraneUpsBeta(h_shell, 50e3, abs(q), mass, radius, grav, 1e3, 1e3, n_max=n_max)
    solver.set_beta(beta)


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
ho = np.concatenate((ho, ho_res))
ho = np.unique(ho)

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
n = np.array(n, dtype=np.float)

E22 = np.zeros(( len(freqs), len(ho)))
E20 = np.zeros(( len(freqs), len(ho)))
E21 = np.zeros(( len(freqs), len(ho)))

Ecrust22 = np.zeros(( len(freqs), len(ho) ))
Ecrust21 = np.zeros(( len(freqs), len(ho) ))
Ecrust20 = np.zeros(( len(freqs), len(ho) ))

EtaTop22 = np.zeros(( len(freqs), len(ho)))
EtaTop21 = np.zeros(( len(freqs), len(ho)))
EtaTop20 = np.zeros(( len(freqs), len(ho)))

# E22_an = np.zeros(len(ho))
# E20_an = np.zeros(len(ho))

DISP = np.zeros(( len(freqs), len(ho) ))

PHI = np.zeros( ( len(ho), len(freqs), 3, 4 ), dtype=np.complex )
PSI = np.zeros( ( len(ho), len(freqs), 3, 4 ), dtype=np.complex )




for i in range(len(ho)):
    for x in [1,-1]:
        ocean_thickness = ho[i]

        # kL, hL, gamma_L = loveNumbers.loading_surface_ocean(den_core, grav, radius, 40e9, nmax=n_max+10)
        # kT, hT, gamma_T = loveNumbers.tidal_surface_ocean(den_core, grav, radius, 40e9)


        solver = LTESolver(rot_rate, radius, grav, x*ocean_thickness, alpha=alpha, nmax=n_max)


        solver.set_beta(beta)

        print("Solving h="+str(ocean_thickness/1e3), end="\r", flush=True)

        cilm = np.zeros((2, n_max+1, n_max+1))

        for j in range(len(freqs)):

            SPRINGS[j] = spring[:4]

            # beta, ups, spring = membraneUpsBeta(h_shell, 50e3, 10*rot_rate, mass, radius, grav, 1e3, 1e3, n_max=15)

            a_20c = coeffs[j][0]
            a_22c = coeffs[j][4]
            a_21c = coeffs[j][2]
            b_21s = coeffs[j][9]
            b_22s = coeffs[j][11]

            forcing_freq = -q_nij[j]

            beta, ups, spring = membraneUpsBeta(h_shell, 50e3, abs(forcing_freq), mass, radius, grav, 1e3, 1e3, n_max=n_max)
            solver.set_beta(beta)

            # --------------------------------------------------------------
            # Solve for libration component, P22
            # if abs(a_22c + b_22s) > 1e-11:
            #     solver.define_forcing( a_22c + b_22s, forcing_freq, 2, 2)

            #     solver.setup_solver()
            #     psi, phi = solver.solve_lte()

            #     if save:
            #         PHI[i, j, 2, :] = phi[:4]
            #         PSI[i, j, 2, :] = psi[:4]

            #     delta1 = np.sum(Nnm2 * (abs(phi)**2.0 + abs(psi)**2.0))

            #     E22[j, i] += delta1

            #     # Ecrust22[j, i] += abs(forcing_freq)/forcing_freq**2.0 * np.sum(Nnm2**2.0 * abs(spring.imag) * abs(phi)**2.0)

            #     Ecrust22[j, i] += np.sum(Nnm2 *n*(n+1) * abs(spring.imag) * abs(phi)**2.0) / abs(forcing_freq)


                # DISP[j, i] += -n*(n+1)/radius**2.0 * 1./forcing_freq * phi.imag

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
                    PHI[i, j, 1, :] = phi[:4]
                    PSI[i, j, 1, :] = psi[:4]

                E21[j, i] += np.sum(Nnm1 * (abs(phi)**2.0 + abs(psi)**2.0))

                # Ecrust21[j, i] += abs(forcing_freq)/forcing_freq**2.0 * np.sum(Nnm1**2.0 * abs(spring.imag) * abs(phi)**2.0)

                Ecrust21[j, i] += np.sum(Nnm1 * n*(n+1) * abs(spring.imag) * abs(phi)**2.0) / abs(forcing_freq)

            # --------------------------------------------------------------
            # Solve for radial component, P20
            #
            # This component is symetrical about q=0, so rather than sum over
            # positive and negative q, we sum only over positive q while
            # doubling the forcing

            # if abs(a_20c) > 1e-11:
            #    solver.define_forcing(a_20c, forcing_freq, 2, 0)
            #    solver.setup_solver()

            #    psi, phi = solver.solve_lte()

            #    if save:
            #        PHI[i, j, 0, :] = phi[:4]
            #        PSI[i, j, 0, :] = psi[:4]

            #    delta2 = np.sum(Nnm0 * (abs(phi)**2.0 + abs(psi)**2.0))
            #    E20[j, i] += delta2

            # #    Ecrust20[j, i] += abs(forcing_freq)/forcing_freq**2.0 * np.sum(n**2 * (n+1)**2 * abs(spring.imag) * abs(phi)**2.0)
            # #    Ecrust20[j, i] += abs(forcing_freq)/forcing_freq**2.0 * np.sum(Nnm0**2.0 * abs(spring.imag) * abs(phi)**2.0)

            #    Ecrust20[j, i] += np.sum(Nnm0 * n*(n+1) * abs(spring.imag) * abs(phi)**2.0) / abs(forcing_freq)

            #    DISP[j, i] += -n*(n+1)/radius**2.0 * 1./forcing_freq * phi.imag


            # if delta1 < 0.0001*E22[i] and delta2 < 0.0001*E20[i]:
            #     break




ho = abs(ho)
# alpha = alpha_actual
E22 *= den * ho *2 *alpha*np.pi
E20 *= den * ho * 2 *alpha*np.pi
E21 *= den * ho * 2 *alpha*np.pi

lambs = 4 * rot_rate**2.0 * radius**2.0 / (grav * ho)
# lambs = 4 *  radius**2.0 / (grav * ho)

Ecrust20 *= 2*np.pi*den*ho**2.0*grav / radius**2
Ecrust22 *= 2*np.pi*den*ho**2.0*grav / radius**2
Ecrust21 *= 2*np.pi*den*ho**2.0*grav / radius**2

# Ecrust20 *= 8*np.pi*den*ho*rot_rate / lambs
# Ecrust22 *= 8*np.pi*den*ho*rot_rate / lambs
# Ecrust21 *= 8*np.pi*den*ho*rot_rate / lambs


if save:
    create_dataset(grp, 'stream function', PSI, np.complex)
    create_dataset(grp, 'velocity potential', PHI, np.complex)
    create_dataset(grp, 'spring constant', SPRINGS, np.complex)
    create_dataset(grp, 'dissipated power: ocean', E20 + E21 + E22, np.float)
    create_dataset(grp, 'dissipated power: shell', Ecrust20 + Ecrust21 + Ecrust22, np.float)

    save_file.close()

# print((E22_an+E20_an)/(E22+E20))

area = 4*np.pi*radius**2.0
fig, ax = plt.subplots(figsize=(4, 3.5))
# ax.loglog(ho/1e3, (np.sum(Ecrust22+Ecrust20+Ecrust21+E22+E20+E21, axis=0)), 'k--', lw=1.0)
ax.loglog(ho/1e3, (np.sum(E22+E20+E21, axis=0)), '-', lw=1.5)
ax.loglog(ho/1e3, (np.sum(Ecrust22+Ecrust20+Ecrust21, axis=0)), '--', lw=1.0)


ax.legend(frameon=False)
ax.set_ylabel("Tidal power [GW]", fontsize=13)
# ax.set_title(forcing_name.capitalize() + " forcing on " + moon_name.capitalize())
ax.grid(which='major', alpha=0.2)
# ax.get_xaxis().get_major_formatter().set_useOffset(False)
# fig = plt.gcf()
fig.savefig("/home/hamish/Dropbox/Tests/"+ save_name + ".pdf", dpi=200, bbox_inches='tight')
plt.show()
