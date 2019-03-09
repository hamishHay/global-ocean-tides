import numpy as np
from scipy import sparse
from scipy.sparse import linalg

class LTEsolver:
    def __init__(self, rot_rate, force_freq, radius, gravity, ho, alpha=0.0, nmax=4, tide_dir='east'):
        self.ho = ho
        self.rot_rate = rot_rate
        self.force_freq = force_freq
        self.radius = radius
        self.gravity = gravity

        self.tide_dir = tide_dir

        self.rot_force = force_freq / (2.*rot_rate)
        self.lambs = 4*rot_rate**2.0 * radius**2.0 / (gravity*ho)
        self.b = alpha/(2. * rot_rate)

        self.nmax = nmax
        self.n = np.arange(1, self.nmax+1, 1, dtype=np.float128)

        self.Kn = self.CalcKn(2)
        self.Ln = self.CalcLn(2)
        self.pn = self.CalcPn(2)
        self.qn = self.CalcQn(2)

        self.CreateLTEMatrix()
        self.CreateForcingVector()
        self.SolveLTE()

        self.CreateResonanceMatrix()
        self.FindResonantThicknesses()

    def CreateResonanceMatrix(self):
        nrows = (self.nmax)
        ncols = (self.nmax)

        coeff_mat = np.zeros((nrows,ncols), dtype=np.complex128)
        for i in range(0, nrows, 1):
            n = int(i+1)
            n_indx = n - 1

            if i-2 >= 0:
                # Left off-diagonal
                coeff_mat[i,i-2] = self.qn[i-2]*self.qn[i-1]/complex(self.b, -self.Ln[i-1])
            if i+2 <= nrows-1:
                # Right off-diagonal
                coeff_mat[i,i+2] = self.pn[i+2]*self.pn[i+1]/complex(self.b, -self.Ln[i+1])

            # Diagonal components
            An = complex(self.b, -self.Ln[i])
            if i > 0 and i < nrows-1:
                An += self.pn[i]*self.qn[i-1]/complex(self.b, -self.Ln[i-1])
                An += self.qn[i]*self.pn[i+1]/complex(self.b, -self.Ln[i+1])

            coeff_mat[i, i] = An
            coeff_mat[i,:] *= -self.rot_force/(n*(n+1))

        self.res_mat = coeff_mat

    def FindResonantThicknesses(self):
        x = sparse.linalg.eigs(self.res_mat, k=40)

        # print(x[0][1::2].imag*4*self.rot_rate**2 * self.radius**2/self.gravity)
        res_thicks = x[0][1::2].imag*4*self.rot_rate**2 * self.radius**2/self.gravity
        print(res_thicks[res_thicks> 0.0])

    def SolveLTE(self):
        x = sparse.linalg.spsolve(self.LTE_mat, self.forcing_vec)

        stream_func_soln = x[::2]
        vel_pot_soln = x[1::2]
        # print(stream_func_soln)
        # print(vel_pot_soln)
        # print(x[3].imag*2*(2+1)/self.radius**2.0*self.ho/self.force_freq)
        # print(x[1].imag*2*(2+1)/self.radius**2.0*self.ho/self.force_freq)


    def CreateForcingVector(self):
        nrows = (self.nmax)*2
        self.forcing_vec = np.zeros(nrows)
        self.forcing_vec[3] = (7./8.)*self.rot_rate**2.0*self.radius**2.0*0.0047

        self.forcing_vec *= 1.0/(2*self.rot_rate)

    def CreateLTEMatrix(self):
        nrows = (self.nmax)*2
        ncols = (self.nmax)*2

        coeff_mat = np.zeros((nrows,ncols), dtype=np.complex128)
        for i in range(0, nrows, 2):
            n = int((i+2)/2)
            n_indx = n - 1

            # Stream function row (even i) #####################################
            if i-1 >= 0:
                coeff_mat[i,i-1] = complex(-self.qn[n_indx - 1], 0) # Left off-diagonal
            if i+3 <= nrows-1:
                coeff_mat[i,i+3] = complex(-self.pn[n_indx + 1], 0) # Right off-diagonal
            coeff_mat[i, i] = complex(self.b, -self.Ln[n_indx])     # diagonal

            # Velocity potential row (odd i) ###################################
            j = i+1
            if j-3 >= 0:
                coeff_mat[j, j-3] = complex(self.qn[n_indx - 1], 0) # Left off-diagonal
            if j+1 <= nrows-1:
                coeff_mat[j, j+1] = complex(self.pn[n_indx + 1], 0) # Right off-diagonal
            coeff_mat[j, j] = complex(self.b, -self.Kn[n_indx])

        self.LTE_mat = sparse.csr_matrix(coeff_mat)

    def CalcKn(self, m):
        n = self.n

        kn = m/(n*(n+1))
        kn += self.rot_force - 1.0*n*(n+1)/(self.lambs*self.rot_force)
        return kn

    def CalcLn(self, m):
        n = self.n
        ln = self.rot_force + m/(n*(n+1))
        return ln

    def CalcPn(self, m):
        n = self.n
        pn = (n+1)*(n+m)/( n*(2*n+1) )
        pn[m-1] = 0.0
        return pn

    def CalcQn(self, m):
        n = self.n
        qn = n*(n+1-m)/( (n+1)*(2*n+1) )
        return qn


if __name__=='__main__':
    solver = LTEsolver(5.31e-5, 5.31e-5, 252.1e3, 0.11, 100, alpha=0.0, nmax=50)
