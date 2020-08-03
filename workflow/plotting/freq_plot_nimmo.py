import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.misc import factorial
import planetPy
import matplotlib.gridspec as gridspec
import sys
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.append("../")

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx}, \\usepackage{libertine}, \\usepackage{libertinust1math}, \\sisetup{detect-all}')

plt.rc('lines', linewidth=0.6)

# matplotlib.rcParams.update({'font.size': 14})
#
# plt.style.use('dark_background')

from LTE_solver import LTESolver

def get_freq_from_day(day, n):
    return (2*np.pi/(day * 24 * 60**2)) / n

def membraneUpsBeta(shell_thickness, ocean_thickness, freq, mass, radius, grav_surf, den_ocean, den_shell, visc_crit, n_max=12):
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


    def visc(T, visc0 = 1e14):
        return visc0 * np.exp(27 * (273/T - 1) )


    def T_profile(Rtop, Rbot, Ttop, Tbot, r):
        d = Rtop - Rbot
        return Tbot**(Rbot/r * (Rtop - r)/d ) * Ttop**(Rtop/r * (r - Rbot)/d )
    # d = np.loadtxt("love_numbers/europa_fluid_love_numbers.dat").T

    # k = d[1, :n_max]
    # h = d[2, :n_max]
    D = np.logspace(0, np.log10(shell_thickness), 5001)

    # D = np.arange(0, shell_thickness+100, 100.0)

    T = T_profile(radius, radius-shell_thickness, 100, 273, radius- D) #np.logspace(100, 273, 10)

    visc_shell = visc(T, visc0 = visc_crit)

    # print(T)
    

    n = np.arange(1, n_max+1, 1)

    # sys.exit()


    G = 6.674e-11

    pois_ratio = 0.33
    rigid_shell = 3.5e9
    # rigid_shell = 60e9
    # rigid_shell  = (3.44 + 0.0001j) * 1e9

    vtop = 1e25

    xx = 0.5


    Te = (1.0 - xx)*shell_thickness
    Tv = (xx)*shell_thickness
    # Tv = shell_thickness

    # Te = 

    # D = np.array([Te, Tv])

    # visc_shell = np.array([vtop, visc_crit])

    

    # print(visc_shell)

    # rigid_shell = (Te*mu_v[0] + Tv*mu_v[1])/shell_thickness

    visc_factor = rigid_shell/(freq*visc_shell)
    mu_v = (rigid_shell / ( 1 -  visc_factor*1j) )

    # Trapezoidal integration

    rigid_shell = np.trapz(mu_v, x=D)/shell_thickness

    p_v = (3*pois_ratio  - (1+pois_ratio)*visc_factor*1j) / (3 - 2*(1 + pois_ratio)*visc_factor*1j)

    pois_ratio = 1./np.trapz( (mu_v / (1 - p_v))  , x=D) * np.trapz( p_v*(mu_v / (1 - p_v)) , x=D)

    # Piecewise integration

    # rigid_shell = (mu_v[0]*Te + mu_v[1]*Tv)/shell_thickness

    # p_v = (3*pois_ratio  - (1+pois_ratio)*visc_factor*1j) / (3 - 2*(1 + pois_ratio)*visc_factor*1j)

    # pois_ratio = 1./((mu_v / (1 - p_v))[0]*Te + (mu_v / (1 - p_v))[1]*Tv ) * ( (p_v*mu_v / (1 - p_v))[0]*Te + (p_v*mu_v / (1 - p_v))[1]*Tv) 

    

    rigid_core = 60e9



    radius_core = radius #- ocean_thickness#(shell_thickness + ocean_thickness)
    radius_ocean = radius - (shell_thickness)

    beta = np.zeros(len(n), dtype=np.complex)
    ups = np.zeros(len(n), dtype=np.complex)


    mass_total = mass


    vol_total = 4./3. * np.pi * radius**3.0


    den_bulk = mass_total/vol_total

    den_ratio= den_ocean/den_bulk


    spring = np.zeros(len(n), dtype=np.complex)
    Xi = np.zeros(len(n))
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
        Xi[l-1] = xi
        



        dsprConst = 1. - (1. + xi*hl)**2.0/(1. + xi*(ht - hl)*sprConst)
        dsprConst *= -sprConst

        dgam_tide = (1. + xi*hl)*ht / (1. + xi*(ht - hl)*sprConst)
        dgam_tide *= -sprConst

        # // Find beta and nu coefficients
        beta[l-1] = 1. - xi*gam_load + sprConst + dsprConst
        ups[l-1] = gam_tide + dgam_tide

        if shell_thickness < 1:
            beta[l-1] = 1. - xi + 0.0 
            ups[l-1] = 1.0
            spring[l-1] = 0.0
        # Assume infinitely rigid mantle
        else:
            beta[l-1] = 1. - xi + sprConst 
            ups[l-1] = 1.0
            spring[l-1] = sprConst

        # print(spring)
    # beta[:] = 1.0

    
    return beta, ups, spring, Xi


moon_name = 'europa'



fmax=80
qmax=60
qmin=-20

data = planetPy.database.Database()
moons = data.jupiter.moons
# moons = data.planet_dict

h_shell = 10

radius = moons[moon_name.capitalize()].radius - h_shell*1e3
grav =  moons[moon_name.capitalize()].gravity
mass = moons[moon_name.capitalize()].mass
den_bulk = moons[moon_name.capitalize()].density
den_ocean = 1000.0

rot_rate = moons[moon_name.capitalize()].mean_motion
alpha = 1e-9


love_file = h5py.File("../shell_love_numbers.h5", 'r')
beta = love_file[moon_name]["beta"][h_shell - 1]
ups = love_file[moon_name]["upsilon"][h_shell - 1][1]

# beta =
beta[0] = 0.00001#0.0001
# beta[:] = 1.0

n_max=30
solver = LTESolver(rot_rate, radius, grav, 1e3, alpha=alpha, nmax=n_max)


conj_freq = rot_rate*0.5
res_list = []
forcing_magnitude = 1
freqs = np.arange(1, fmax, 0.5)


mode_num = 5
res_m2_east = np.zeros((mode_num, len(freqs)))
res_m0 = np.zeros((mode_num, len(freqs)))
res_m2_west = np.zeros((mode_num, len(freqs)))

i=0
for q in freqs:
    forcing_freq = (conj_freq)*q

    beta, ups, spring, xi = membraneUpsBeta(10e3, 50e3, abs(forcing_freq), mass, radius, grav, 1e3, 1e3, 1e17, n_max=n_max)
    beta[0] = 0.000001
    solver.set_beta(beta)

    print(forcing_freq, beta[1])

    solver.define_forcing(forcing_magnitude, forcing_freq, 2, 0)
    solver.setup_solver()
    resH = solver.find_resonant_thicknesses()[1:]

    resH = resH[resH>0.0]
    for k in range(mode_num):
        try:
            res_m0[k, i] = resH[k]
        except IndexError:
            res_m0[k, i] = np.nan


    solver.define_forcing(forcing_magnitude, -forcing_freq, 2, 2)
    solver.setup_solver()
    resH = solver.find_resonant_thicknesses()[1:]
    resH = resH[resH>0.0]
    for k in range(mode_num):
        try:
            res_m2_west[k, i] = resH[k]
        except IndexError:
            res_m2_west[k, i] = np.nan


    solver.define_forcing(forcing_magnitude, forcing_freq, 2, 2)
    solver.setup_solver()
    resH = solver.find_resonant_thicknesses()[1:]
    resH = resH[resH>0.0]

    for k in range(mode_num):
        try:
            res_m2_east[k, i] = resH[k]
        except IndexError:
            res_m2_east[k, i] = np.nan

    i+=1


fig = plt.figure(figsize=(8,9), constrained_layout=False)

gs = gridspec.GridSpec(3, 3, fig)

gs.update(wspace=0.0, hspace=0.3)
# fig, ax = plt.subplots()
# ax.contourf(ho, freqs, np.log10(data1.T))

ax1 = plt.subplot(gs[:2,:2])
ax2 = plt.subplot(gs[:2, 2])
ax4 = plt.subplot(gs[2, :2])
# ax5 = plt.subplot(gs[2, 2])

nr_thickness = np.zeros(len(freqs))
for i in range(len(freqs)):
    beta, ups, spring, xi = membraneUpsBeta(10e3, 50e3, abs(freqs[i]*conj_freq), mass, radius, grav, 1e3, 1e3, 1e17, n_max=n_max)
    qw = abs(freqs[i]*conj_freq)**2.0 * radius / grav
    nr_thickness[i] = qw * radius / (2*(2+1)) / (1 - xi[1] + spring[1].real)

freqs /= 2

ax1.plot(nr_thickness/1e3, freqs, color="#ff6666", lw=1.2, alpha=0.85)
ax1.plot(nr_thickness/1e3, -freqs, color="#ff6666", lw=1.2, alpha=0.85)


h_range = np.logspace(2, np.log10(150e3), 1001)

# a = 1 
# b = 1/3. * rot_rate
# c = -6 * grav * h_range/ radius**2.0
# res_freq_east = (-b + np.sqrt(b**2.0 - 4 * a * c))/ (2*a) 
# res_freq_west = (-b - np.sqrt(b**2.0 - 4 * a * c))/ (2*a) 

# ax1.plot(h_range/1e3, -res_freq_west/rot_rate, 'b')

th2 = ax1.text(0.5, -2.1, 'non-rotating limit', fontsize=8, rotation=12, color="#ff6666", alpha=0.85)

lines = ['-', '--', '-.', ':']
ii = 0
for i in range(mode_num):

    if i%2 == 0:
        if i==0:
            l0 = "$m=0$"
            l1 = "$m=2$"
        else:
            l0 = None
            l1 = None
        ax1.plot(res_m2_east[i]/1e3, freqs, lines[0], color='k', lw=(1.0-ii*0.3))
        ax1.plot(res_m2_west[i]/1e3, -freqs, lines[0], color='k', lw=(1.0-ii*0.3), label=l1)
        ax1.plot(res_m0[i]/1e3, freqs, lines[1], color='k', lw=(1.0-ii*0.3), label=l0)
        ax1.plot(res_m0[i]/1e3, -freqs, lines[1], color='k', lw=(1.0-ii*0.3))

        ii+=1





datafile = h5py.File("FigureData.h5", 'a')

print(np.shape(freqs), np.shape(res_m2_east.T))


grp = datafile.create_group("Figure 1a")

d = np.concatenate((freqs[:, None], res_m2_east.T), axis=1)
grp.create_dataset("m2 resonances east", data=d)

d = np.concatenate((freqs[:, None], res_m0.T), axis=1)
grp.create_dataset("m0 resonances", data=d)

d = np.concatenate((-freqs[:, None], res_m2_west.T), axis=1)
grp.create_dataset("m2 resonances west", data=d)

grp.attrs["Explanation"] = "Column 1 = frequency (y-axis), column 2-4 = resonant ocean thickness"


ax1.grid(which='major', alpha=0.4)

#
ax1.set_xscale("log")
ax1.set_xlim([0.1,150])

# ax1.set_ylim([-qmin/2,-qmax/2][::-1])



ax1.invert_yaxis()

ax1.set_ylim([-qmin/2,-qmax/2])

ax1.legend(frameon=False, prop={'size': 9})


ax1.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))



load_file = h5py.File('galilean_coefficients.h5','r')

tc_io = np.array(load_file['europa']['io']['tide coeffs'][:, :])
tc_gan = np.array(load_file['europa']['ganymede']['tide coeffs'][:, :])
tc_jup = np.array(load_file['europa']['eccentricity']['tide coeffs'][:, :])
freqs  = np.array(load_file['europa']['eccentricity']['frequency q'][:])/2
qs = np.arange(0, len(tc_io[:,0]))

tc_total = tc_jup + tc_io + tc_gan

# freqs = np.arange(-100, 100, 1)

max_c = 1

# ax2.fill_between(np.log10(abs(tc_jup[:, 4]+tc_jup[:, 11])), freqs-0.25, step="pre", alpha=0.6, label='Jupiter')
ax2.fill_between(np.log10(abs(tc_jup[:, 4]+tc_jup[:, 11])), freqs-0.25, step="pre", alpha=0.6, label='Jupiter')
# ax2.plot(abs(tc_jup[:, 4]+tc_jup[:, 11]), freqs-0.25-0.25, drawstyle="steps", lw=0.6)

ax2.fill_between(np.log10(abs(tc_io[:, 4]+tc_io[:, 11])), freqs-0.25, step="pre", alpha=0.6, label='Io')
# # ax2.plot(abs(tc_io[:, 4]+tc_io[:, 11]), freqs-0.25-0.25, drawstyle="steps", lw=0.6, label="Io", alpha=0.0)
#
ax2.fill_between(np.log10(abs(tc_gan[:, 4]+tc_gan[:, 11])), freqs-0.25, step="pre", alpha=0.6, label='Ganymede')
# # ax2.plot(abs(tc_gan[:, 4]+tc_gan[:, 11]), freqs-0.25-0.25, drawstyle="steps", lw=0.6, label='Ganymede', alpha=0.0)
#
# # ax2.fill_between(abs(tc_total[:, 4]+tc_total[:, 11]), freqs-0.25-0.25, step="pre", alpha=0.6, label='Total')
#
ax2.plot(np.log10(abs(tc_total[:, 4]+tc_total[:, 11])), freqs-0.25, 'k', drawstyle="steps", lw=0.9, label='Total')
# ax2.plot(x,y2, drawstyle="steps")

ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
# print(tc_gan[90:110, 4]+tc_gan[90:110, 11]+ tc_io[90:110, 4]+ tc_io[90:110, 11]+tc_jup[90:110, 4]+tc_jup[90:110, 11])


grp = datafile.create_group("Figure 1b")

d = np.vstack( (freqs.T, tc_jup[:, 4].T+tc_jup[:, 11].T) ).T
grp.create_dataset("jupiter forcing", data=d)

d = np.vstack( (freqs.T, tc_io[:, 4].T+tc_io[:, 11].T) ).T
grp.create_dataset("io forcing", data=d)

d = np.vstack( (freqs.T, tc_gan[:, 4].T+tc_gan[:, 11].T) ).T
grp.create_dataset("ganymede forcing", data=d)

grp.attrs["explanation"] = "Column 1 = frequency (y-axis), column 2 = m2 forcing"

ax3 = ax2.twinx()
# ax3.spines["right"].set_position(("axes", 1.05))
# ax3.set_ylim([qmin/2, qmax/2])
ax3.set_ylabel("Forcing period [days]")

days = np.array([-0.4, -0.8, -3.5, 3.5, 0.8, 0.4, 0.25, 0.2, 0.15])*-1

ax3.invert_yaxis()

ax3.set_ylim([-qmin/2,-qmax/2])

ax3.set_yticks(get_freq_from_day(days, conj_freq*2))

days_text = [str(i) for i in days]

ax3.set_yticklabels(days_text)


ax5 = inset_axes(ax4, 1.7, 1.6, bbox_to_anchor=(0.9, 0.31),bbox_transform=ax4.figure.transFigure)
# infile = h5py.File('plot_diss.h5', 'r')

# diss = infile['europa']['eccentricity']['dissipated power'][:,:]
# diss_ecc = diss[98, :]
# diss_io = diss[120, :]

# ho = infile['europa']['eccentricity']['ocean thickness'][:]/1e3

grp = datafile.create_group("Figure 1c")



# d = np.concatenate( (freqs, tc_io[:, 4]+tc_io[:, 11]) )
# grp.create_dataset("io forcing", data=d)

# d = np.concatenate( (freqs, tc_gan[:, 4]+tc_gan[:, 11]) )
# grp.create_dataset("ganymede forcing", data=d)

# 

# infile = h5py.File('plot_diss_ecc_a1e-9.h5', 'r')
infile = h5py.File('plot1_diss_eur_v1e17_a1e-9.h5', 'r')
f = infile['europa']['frequency'][:]
q1_idx = list(f).index(-2)
q10_idx = list(f).index(20)

diss_ecc = infile['europa']['dissipated power: ocean'][q1_idx,:] + infile['europa']['dissipated power: shell'][q1_idx,:]
diss_io = infile['europa']['dissipated power: ocean'][q10_idx,:] + infile['europa']['dissipated power: shell'][q10_idx,:]

ho = infile['europa']['ocean thickness'][:]/1e3

d = np.vstack( (ho.T, diss_ecc.T, diss_io.T) ).T
print(d)
grp.create_dataset("dissipated power alpha=1e-9", data=d)

infile.close()

ax4.loglog(ho, diss_ecc, c='C0', lw=1.4)
ax4.loglog(ho, diss_io, c='C1', lw=1.4)


infile = h5py.File('plot1_diss_eur_v1e17_a1e-6.h5', 'r')
f = infile['europa']['frequency'][:]
q1_idx = list(f).index(-2)
q10_idx = list(f).index(20)

diss_ecc = infile['europa']['dissipated power: ocean'][q1_idx,:] + infile['europa']['dissipated power: shell'][q1_idx,:]
diss_io = infile['europa']['dissipated power: ocean'][q10_idx,:] + infile['europa']['dissipated power: shell'][q10_idx,:]

ho = infile['europa']['ocean thickness'][:]/1e3

d = np.vstack( (ho.T, diss_ecc.T, diss_io.T) ).T
grp.create_dataset("dissipated power alpha=1e-6", data=d)

grp.attrs["explanation"] = "Column 1 = ocean thickness (x-axis), column 2 = jupiter forcing, column 3 = io forcing"

infile.close()

ax4.loglog(ho, diss_ecc,'k:', dashes=(0.8, 0.8), lw=1.4)
ax4.loglog(ho, diss_io,'k:', dashes=(0.8, 0.8), lw=1.4)

# infile = h5py.File('plot_diss2.h5', 'r')
# diss = infile['europa']['dissipated power'][:,:]
# f = infile['europa']['frequency'][:]

# q1_idx = list(f).index(-2)
# q10_idx = list(f).index(20)

# diss_ecc = diss[q1_idx, :]
# diss_io = diss[q10_idx, :]

# ho = infile['europa']['ocean thickness'][:]/1e3

infile = h5py.File('plot1_zoom_diss_eur_v1e17_a1e-9.h5', 'r')
f = infile['europa']['frequency'][:]
q1_idx = list(f).index(-2)
q10_idx = list(f).index(20)

diss_ecc = infile['europa']['dissipated power: ocean'][q1_idx,:] + infile['europa']['dissipated power: shell'][q1_idx,:]
diss_io = infile['europa']['dissipated power: ocean'][q10_idx,:] + infile['europa']['dissipated power: shell'][q10_idx,:]

ho = infile['europa']['ocean thickness'][:]/1e3

infile.close()

ax5.semilogy(ho, diss_ecc, c='C0', lw=1.4)
ax5.semilogy(ho, diss_io, c='C1', lw=1.4)

infile = h5py.File('plot1_zoom_diss_eur_v1e17_a1e-6.h5', 'r')
f = infile['europa']['frequency'][:]
q1_idx = list(f).index(-2)
q10_idx = list(f).index(20)

diss_ecc = infile['europa']['dissipated power: ocean'][q1_idx,:] + infile['europa']['dissipated power: shell'][q1_idx,:]
diss_io = infile['europa']['dissipated power: ocean'][q10_idx,:] + infile['europa']['dissipated power: shell'][q10_idx,:]

ho = infile['europa']['ocean thickness'][:]/1e3

infile.close()

ax5.semilogy(ho, diss_ecc, 'k:', dashes=(0.8, 0.8), lw=1.4)#, c='C1')
ax5.semilogy(ho, diss_io, 'k:', dashes=(0.8, 0.8), lw=1.4)#, c='C0')

# ax5.semilogy(ho, diss_ecc, c='C0', lw=1.4)
# ax5.semilogy(ho, diss_io, c='C1', lw=1.4)

# infile = h5py.File('plot_diss_a6.h5', 'r')

# diss = infile['europa']['eccentricity']['dissipated power'][:,:]
# diss_ecc = diss[98, :]
# diss_io = diss[120, :]

# ho = infile['europa']['eccentricity']['ocean thickness'][:]/1e3

# ax4.loglog(ho, diss_ecc, 'k:', dashes=(0.8, 0.8), lw=1.4)#, c='C1')
# ax4.loglog(ho, diss_io, 'k:', dashes=(0.8, 0.8), lw=1.4)#, c='C0')

# infile = h5py.File('plot_diss2_a6.h5', 'r')
# diss = infile['europa']['dissipated power'][:,:]
# f = infile['europa']['frequency'][:]

# ho = infile['europa']['ocean thickness'][:]/1e3

# q1_idx = list(f).index(-2)
# q10_idx = list(f).index(20)

# diss_ecc = diss[q1_idx, :]
# diss_io = diss[q10_idx, :]




ax4.set_xlim([0.1, 150])
ax5.set_xlim([14.5,16.0])
ax5.set_ylim([1e5,1e14])


ax2.set_ylim([-fmax,fmax])
ax2.yaxis.set_ticklabels([])
ax2.legend(frameon=False, prop={'size': 9})

ax2.grid(which='major', alpha=0.4)
#
#
ax1.set_ylabel("Forcing frequency, $q$")
# ax1.tick_params(labelbottom=False)
ax1.set_xlabel("Resonant ocean thickness [km]")
ax4.set_xlabel("Ocean thickness [km]")
ax5.set_xlabel("Ocean thickness [km]")

# ax2.yaxis.tick_right()
#
# ax2.set_xscale("log")
ax2.set_xlim([-5, 2])
ax2.set_ylim([qmin/2,qmax/2])
ax2.set_xlabel("log$_{10}$(Tidal potential [m$^2$ s$^{-2}$])")

locmaj = matplotlib.ticker.LogLocator(base=10,numticks=40)
ax4.yaxis.set_minor_locator(locmaj)
ax4.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax4.grid(which='major', alpha=0.4)

locmaj = matplotlib.ticker.LogLocator(base=10,numticks=40)
ax5.yaxis.set_minor_locator(locmaj)
ax5.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax5.yaxis.tick_right()
ax5.yaxis.set_label_position("right")
ax5.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

ax4.set_ylabel("Tidal power [W]")
ax5.set_ylabel("Tidal power [W]")

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax4, ax5, loc1=2, loc2=3, fc="none", ec="0.7")
# ax4.indicate_inset_zoom(ax5, loc1=1)

# ax1.text(0.2, 4e2, '$q=10$, $\\alpha=10^{-9}$', fontsize=7, rotation=4)
# ax1.text(0.2, 4e5, '$q=10$, $\\alpha=10^{-6}$', fontsize=7, rotation=3.5)
ax1.text(20, -16, 'mode 1', fontsize=8, rotation=52)
ax1.text(5.6, -16, 'mode 2', fontsize=8, rotation=53)
ax1.text(2.6, -16, 'mode 3', fontsize=8, rotation=53)

labels = ["{:d}".format(int(item)) for item in ax1.get_yticks().tolist()]
ax1.set_yticklabels(labels)


ax4.text(0.2, 4e2, '$q=-10$, $\\alpha=10^{-9}$ \\si{\\per\\second}', fontsize=7, rotation=4)
ax4.text(0.2, 4e5, '$q=-10$, $\\alpha=10^{-6}$ \\si{\\per\\second}', fontsize=7, rotation=3.5)
ax4.text(0.75, 6e9, '$q=1$, $\\alpha=10^{-9}$ \\si{\\per\\second}', fontsize=7, rotation=-4.5)
ax4.text(0.75, 1e13, '$q=1$, $\\alpha=10^{-6}$ \\si{\\per\\second}', fontsize=7, rotation=-4.5)

ax4.set_xticklabels([0.1, 0.1, 1, 10, 100], fontsize=10)
ax1.set_xticklabels([0.1, 0.1, 1, 10, 100], fontsize=10)
ax5.set_xticklabels([14.5, 15, 15.5, 16], fontsize=10)

labels = ["\\num[retain-unity-mantissa = false]{{{:1.0e}}}".format(item) for item in ax4.get_yticks().tolist()]
ax4.set_yticklabels(labels)#, fontsize=ticksize)

tt = ax1.text(-0.12, 1.0, "\\textbf{a}", fontfamily="sans-serif", transform=ax1.transAxes,
            size=13, weight='bold')

tt = ax2.text(1.1, 1.0, "\\textbf{b}", fontfamily="sans-serif", transform=ax2.transAxes,
            size=13, weight='bold')

tt = ax4.text(-0.12, 1.0, "\\textbf{c}", fontfamily="sans-serif", transform=ax4.transAxes,
            size=13, weight='bold')

# tt = ax5.text(1.1, 1.0, "\\textbf{d}", fontfamily="sans-serif", transform=ax5.transAxes,
#             size=13, weight='bold')

# fig.suptitle("Europan tides and response, 20 km shell", y=0.915)
# fig = plt.gcf()
fig.savefig("/home/hamish/Dropbox/Tests/europa_freqs.pdf", format='pdf', bbox_inches='tight')#, transparent=True)
# plt.show()





# ax1.plot(res_m2_east[i]/1e3, -freqs, lines[0], color='k', lw=(1.0-ii*0.3))
# ax1.plot(res_m2_west[i]/1e3, freqs, lines[0], color='k', lw=(1.0-ii*0.3), label=l1)
# ax1.plot(res_m0[i]/1e3, freqs, lines[1], color='k', lw=(1.0-ii*0.3), label=l0)
# ax1.plot(res_m0[i]/1e3, -freqs, lines[1], color='k', lw=(1.0-ii*0.3))

# plt.savefig("")


datafile.close()