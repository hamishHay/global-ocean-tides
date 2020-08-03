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

def membraneUpsBeta(shell_thickness, ocean_thickness, freq, mass, radius, grav_surf, den_ocean, den_shell, visc_crit, t_bottom=273, Ea=60e3, n_max=12):
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
        Tref = 273
        # return
        R = 8.314 
        # Ea = 60e3

        # Ea = 60e3 * np.ones(len(T))
        # Ea[T > 258] = 180e3
        return visc0 * np.exp(Ea/ (R*Tref) * (Tref/T -1 ) ) #np.exp(27 * (Tref/T - 1) )


    def T_profile(Rtop, Rbot, Ttop, Tbot, r):
        d = Rtop - Rbot
        return Tbot**(Rbot/r * (Rtop - r)/d ) * Ttop**(Rtop/r * (r - Rbot)/d )
    # d = np.loadtxt("love_numbers/europa_fluid_love_numbers.dat").T

    # k = d[1, :n_max]
    # h = d[2, :n_max]
    D = np.logspace(0, np.log10(shell_thickness), 5001)

    T = T_profile(radius, radius-shell_thickness, 100, t_bottom, radius- D) #np.logspace(100, 273, 10)

    print(T)
    

    n = np.arange(1, n_max+1, 1)

    # sys.exit()


    G = 6.674e-11

    pois_ratio = 0.33
    rigid_shell = 3.5e9
    # rigid_shell = 60e9
    # rigid_shell  = (3.44 + 0.0001j) * 1e9

    vtop = 1e19
    vbot = 1e16

    xx = 0.0

    Te = xx*shell_thickness
    Tv = (1 - xx)*shell_thickness
    Tv = shell_thickness

    # D = np.array([Te, Tv])

    visc_shell = np.array([vtop, vbot])

    visc_shell = visc(T, visc0 = visc_crit)

    # rigid_shell = (Te*mu_v[0] + Tv*mu_v[1])/shell_thickness

    visc_factor = rigid_shell/(freq*visc_shell)
    mu_v = (rigid_shell / ( 1 -  visc_factor*1j) )

    

    rigid_shell = np.trapz(mu_v, x=D)/shell_thickness

    p_v = (3*pois_ratio  - (1+pois_ratio)*visc_factor*1j) / (3 - 2*(1 + pois_ratio)*visc_factor*1j)
    
    pois_ratio = (3*pois_ratio  - (1+pois_ratio)*visc_factor*1j) / (3 - 2*(1 + pois_ratio)*visc_factor*1j)

    pois_ratio = 1./np.trapz( (mu_v / (1 - p_v))  , x=D) * np.trapz( p_v*(mu_v / (1 - p_v)) , x=D)

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

    
    return beta, ups, spring, visc_shell







data = planetPy.database.Database()
moons = data.moon_dict
# moons = data.planet_dict

moon_name = 'europa'


# print(save_name)
rot_rate = moons[moon_name.capitalize()].mean_motion


radius = moons[moon_name.capitalize()].radius# - h_shell*1e3
grav =  moons[moon_name.capitalize()].gravity
den_bulk = moons[moon_name.capitalize()].density
# den_ocean = 1000.0
mass = moons[moon_name.capitalize()].mass
ecc = moons[moon_name.capitalize()].eccentricity

h_shell=10e3
den = 1e3
n_max = 3
viscosities = [1e16, 1e18]
tbot = np.array([270, 260, 250])


fig, ax = plt.subplots() 

Ea = [90e3, 60e3, 30e3]

linestyles = ['-', '--', '-.']
for j in range(len(viscosities)):
    ls = linestyles[j]
    for E in Ea:
        imag_bit = np.zeros(len(tbot))

        beta, ups, spring, v_profile = membraneUpsBeta(h_shell, 10e3, 10*rot_rate, mass, radius, grav, den, den, viscosities[j], 273, E, n_max=n_max)

        # for i in range(len(tbot)):
        #     beta, ups, spring, v_profile = membraneUpsBeta(h_shell, 10e3, 10*rot_rate, mass, radius, grav, den, den, viscosities[j], tbot[i], E, n_max=n_max)


        #     imag_bit[i] = beta[1].imag
            # print(beta[1].imag)



        ax.semilogy(v_profile, ls=ls)

fig.savefig("/home/hamish/Dropbox/Tests/imag.pdf")