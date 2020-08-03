import numpy as np
from scipy.special import factorial

def eff_rigidity(mu, den, g, R, degree=2):
    return (2*degree**2. + 4*degree + 3) / degree * mu / (den*g*R)

def tidal_surface_ocean(den_solid, g, R, mu, degree=2):
    mu_bar = eff_rigidity(mu, den_solid, g, R, degree=degree)

    k2 = 1.0/(1 + mu_bar) * 3./(2.*(degree - 1))
    h2 = 1.0/(1 + mu_bar) * (2*degree + 1)/(2*(degree - 1))

    gamma_T = (1 + k2 - h2)

    return k2, h2, gamma_T

def loading_surface_ocean(den_solid, g, R, mu, nmax=12, den_fluid=1000.0):
    degree = np.arange(2, nmax+1, 1)

    mu_bar = eff_rigidity(mu, den_solid, g, R, degree=degree)
    kL = -1.0/(1 + mu_bar) * 1.0
    hL = -1.0/(1 + mu_bar) * (2*degree + 1)/3

    gamma_L = 1 - (1+kL-hL)*3*den_fluid/((2*degree+1)*den_solid)

    return kL, hL, gamma_L

def k2_maxwell(den, g, R, mu, eta, freq):
    rigid_eff = 19./2. * mu/(den*g*R)

    maxwell_time = eta/mu

    k_im = -1.5 * rigid_eff * freq* maxwell_time
    k_im /= (1 + (1+rigid_eff)**2. * (freq*maxwell_time)**2.0)

    return k_im

def k2_burgers(den, g, R, mu, etas, freq):
    Ju = 1./mu#1.66e-11
    dJ = 0.2*Ju
    etap = 0.02*etas

    rigid_eff = 19./2. * mu/(den*g*R)

    lam = (Ju*etap*freq)**2. + (Ju/dJ)**2.

    k_im  = (dJ*etas*freq)**-2. + lam*(rigid_eff+1)**2.
    k_im += 2*Ju/dJ * (rigid_eff + 1) + (etap/etas+1)**2.0
    k_im  = 1./k_im
    k_im *= ((dJ*etas*freq)**-2. + (etap/etas)**2. + etap/etas)
    k_im *= -1.5 * Ju*etas*freq*rigid_eff

    return k_im

def k2_andrade(den, g, R, mu, etas, freq):
    Ju = 1./mu
    a = 0.20
    xi = 1.0

    S = factorial(a)*np.sin(a/2 * np.pi)
    C = factorial(a)*np.cos(a/2 * np.pi)

    rigid_eff = 19./2. * mu/(den*g*R)

    k_im  = 2.*(Ju*etas*freq)**(2.-a) * xi**-a * (S/(Ju*etas*freq) + (rigid_eff+1)*C)
    k_im += (Ju*etas*freq)**(2.*(1-a))*factorial(a)**2.0
    k_im += 1 + (Ju*etas*freq)**2.0 * (rigid_eff+1)**2.0
    k_im  = 1./k_im
    k_im *= (1 + (Ju*etas*freq)**(1.-a) *xi**-a * S)
    k_im *= -1.5 * Ju*etas*freq*rigid_eff

    return k_im

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

    
    return beta, ups, spring
