import numpy as np
from scipy import sparse
from scipy.sparse import linalg

class LTESolver:
    """Class to semianalytically solve the global Laplace Tidal Equations

    This class contains the relevant methods to solve the Laplace Tidal
    Equations in a global ocean for a given set of input parameters and tidal
    forcing. The solution method used was originally developed by Longuet and
    Higgins (1966), although this code is based on Matsuyama et al., (2018)
    and Beuthe (2016).

    Attributes:
        rot_rate: rotation rate of the tidally deformed body [rad s^-1].
        ho: thickness of the ocean [m].
        radius: mean radius of the body [m].
        gravity: surface gravity at the mean radius of the body [m s^-2].
        alpha: drag coefficient [s^-1]

        lambs: Lamb's parameter (Matsuyama et al, 2018, Eq. C.7)
        rot_force: Ratio of rotation freq to forcing freq (Matsuyama et al,
                   2018, Eq. C.7)
        b: Tidal Q (Matsuyama et al, 2018, Eq. C.6)

        Kn, Ln, pn, qn: Spherical harmonic constants (Matsuyama et al, 2018,
                        Eq. C.6)

        nmax: Maximum spherical harmonic degree for the calculations
        n: array of spherical harmonics degrees, starting from degree=1.

    """

    def __init__(self, rot_rate, radius, gravity, ho, alpha=0.0, nmax=4):
        """Define problem parameters"""
        self.ho = ho
        self.rot_rate = rot_rate
        self.radius = radius
        self.gravity = gravity

        self.lambs = 4*rot_rate**2.0 * radius**2.0 / (gravity*ho)
        self.b = alpha/(2. * rot_rate)

        self.nmax = nmax
        self.n = np.arange(1, self.nmax+1, 1, dtype=np.float128)
        self.m = None


    def setup_solver(self):
        """Setup all constants and matrices to solve the linear system Ax=b"""
        if self.m is not None:
            self.Kn = self.calc_kn(self.m)
            self.Ln = self.calc_ln(self.m)
            self.pn = self.calc_pn(self.m)
            self.qn = self.calc_qn(self.m)

        else:
            raise ValueError("Forcing potential has not been defined. Run define_forcing first!")

        self.create_lte_matrix()


    def create_resonance_matrix(self):
        """Define matrix to use in an eigenvalue problem"""
        nrows = self.nmax
        ncols = self.nmax

        coeff_mat = np.zeros((nrows,ncols), dtype=np.complex128)
        for i in range(0, nrows, 1):
            n = int(i+1)
            n_indx = n - 1

            # Left off-diagonal
            if i-2 >= 0:
                coeff_mat[i,i-2] = self.qn[i-2]*self.qn[i-1]/complex(self.b, -self.Ln[i-1])

            # Right off-diagonal
            if i+2 <= nrows-1:
                coeff_mat[i,i+2] = self.pn[i+2]*self.pn[i+1]/complex(self.b, -self.Ln[i+1])

            # Diagonal components
            An = complex(self.b, -self.Ln[i])
            if i > 0 and i < nrows-1:
                An += self.pn[i]*self.qn[i-1]/complex(self.b, -self.Ln[i-1])
                An += self.qn[i]*self.pn[i+1]/complex(self.b, -self.Ln[i+1])

            coeff_mat[i, i] = An
            coeff_mat[i,:] *= -self.rot_force/(n*(n+1))

        return sparse.csr_matrix(coeff_mat)


    def find_resonant_thicknesses(self):
        """Find the eigenvalues of the LTE coefficient matrix"""
        self.res_mat = self.create_resonance_matrix()

        # Get the eigenvalues (1/lambs)
        x = sparse.linalg.eigs(self.res_mat, k=self.nmax-5)

        res_thicks = x[0][1::2].imag*4*self.rot_rate**2 * self.radius**2/self.gravity

        return res_thicks


    def solve_lte(self):
        """Solve the Laplace Tidal Equations using the coefficient matrix LTE_mat"""
        x = sparse.linalg.spsolve(self.LTE_mat, self.forcing_vec)

        stream_func_soln = x[::2]
        vel_pot_soln = x[1::2]

        return stream_func_soln, vel_pot_soln


    def define_forcing(self, magnitude, freq, degree, order):
        """Create the vector b in Ax=b using a user defined forcing potential.

        Creates a vector of zeros except for the forcing potential at the input
        degree. All constants in the problem will be defined for this degree and
        order.

        Args:
            magnitude: magnitude of the forcing potential
            freq: frequency of the forcing potential
            degree: spherical harmonic degree of the forcing potential, n
            order: spherical harmonic order of the forcing potential, m

        """

        nrows = self.nmax*2
        self.forcing_vec = np.zeros(nrows)
        self.forcing_vec[degree+1] = magnitude
        self.forcing_vec *= 1.0/(2*self.rot_rate)

        self.force_freq = freq
        self.rot_force = self.force_freq / (2.*self.rot_rate)
        self.m = order


    def create_lte_matrix(self):
        """Create the matrix A of LTE coefficients in Ax=b"""
        nrows = self.nmax*2
        ncols = self.nmax*2

        coeff_mat = np.zeros((nrows,ncols), dtype=np.complex128)
        for i in range(0, nrows, 2):
            n = int((i+2)/2)
            n_indx = n - 1

            # Stream function row (even i)
            if i-1 >= 0:
                coeff_mat[i,i-1] = complex(-self.qn[n_indx - 1], 0) # Left off-diagonal
            if i+3 <= nrows-1:
                coeff_mat[i,i+3] = complex(-self.pn[n_indx + 1], 0) # Right off-diagonal
            coeff_mat[i, i] = complex(self.b, -self.Ln[n_indx])     # diagonal

            # Velocity potential row (odd i)
            j = i+1
            if j-3 >= 0:
                coeff_mat[j, j-3] = complex(self.qn[n_indx - 1], 0) # Left off-diagonal
            if j+1 <= nrows-1:
                coeff_mat[j, j+1] = complex(self.pn[n_indx + 1], 0) # Right off-diagonal
            coeff_mat[j, j] = complex(self.b, -self.Kn[n_indx])

        # Convert from dense to sparse matrix format
        self.LTE_mat = sparse.csr_matrix(coeff_mat)

    def calc_kn(self, m):
        n = self.n
        kn = m/(n*(n+1))
        kn += self.rot_force - 1.0*n*(n+1)/(self.lambs*self.rot_force)
        return kn

    def calc_ln(self, m):
        n = self.n
        ln = self.rot_force + m/(n*(n+1))
        return ln

    def calc_pn(self, m):
        n = self.n
        pn = (n+1)*(n+m)/( n*(2*n+1) )
        pn[m-1] = 0.0
        return pn

    def calc_qn(self, m):
        n = self.n
        qn = n*(n+1-m)/( (n+1)*(2*n+1) )
        return qn


if __name__=='__main__':
    rot_rate = 5.31e-5
    radius = 252.1e3
    grav = 0.11
    ocean_thickness = 1e3

    forcing_magnitude = -(1./8.)*rot_rate**2.0*(radius)**2.0*0.0047

    solver = LTESolver(rot_rate, radius, grav, ocean_thickness, alpha=1e-8, nmax=80)

    solver.define_forcing(forcing_magnitude, -rot_rate, 2, 2)
    solver.setup_solver()


    psi, phi = solver.solve_lte()
    resH = solver.find_resonant_thicknesses()
