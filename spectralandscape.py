import numpy as np
import GPy

def construct_Wigner(R_ix):
    ###### create: random-Matrix à la Wigner
    l       = int( 0.5*(np.sqrt(8*R_ix.shape[1]+1)-1) )
    M       = np.zeros((R_ix.shape[0], l, l), dtype=R_ix.dtype)
    i, j    = np.triu_indices(l, k=0)
    for n in range(R_ix.shape[0]):
        M[n, i, j] = R_ix[n] ## wrap this into a symmetric matrix
        M[n, j, i] = R_ix[n]
    return M

class SpectraLandscape():
    """ built: 3/27/2024
    this an object of a Random-Contionus-Function (RCF), with-respect-to a gpy kernel
    RCF : IN -> OUT = R^(d)
    we define a prior, and then sample to form a posterior.
    """

    def __init__(self, Domain:np.ndarray, N:int, d:int=100, seed:int=777,
                 l:int=10, d_min:np.float64=0, d_max:np.float64=1., γ:np.float64=0.0,
                 IN_noise=None, OUT_noise=None,
                 kernel=GPy.kern.RBF):
        """ !! note datatypes should be tf.float64 for stable Cholesky-operations
        GIVEN >
             Domain : 2d-np.ndarray (with shape=(d,2), with d=# of dims )
                  N : int (number-of-defining-points)
                 d : int (Multiple-Output Dimension)
             **seed : int
           **kernel : GPy.kern
         **IN_noise : 1d-np.ndarray (len == Domain.shape[1])
        **OUT_noise : 1d-np.ndarray (len == d)

        GET   >
            None
        """

        self.dtype  = np.float64
        self.IN     = Domain.astype(self.dtype)  ### : np.ndarray (IN-space range)
        self.N      = N      ### number of defining points
        self.d      = d     ### int (dimension of OUT)
        self.l      = l    ### : inter (features in the spectra)
        self.kernel = kernel(self.IN.shape[0])
        self.seed   = seed ### define pseudo-random seed
        self.d_min  = d_min
        self.d_max  = d_max
        self.γ      = γ

        self.MO     = self.l*(self.l+1)//2

        np.random.seed( self.seed )

        ### define anisotropic i.i.d white-noise
        if IN_noise is None:
            self.IN_noise=np.zeros(self.IN.shape[0], dtype=self.dtype)
        else:
            self.IN_noise = IN_noise
        if OUT_noise is None:
            self.OUT_noise=np.zeros(self.MO, dtype=self.dtype)
        else:
            self.OUT_noise = OUT_noise

        ### define IN-space defining-points
        self.R_ix  = np.random.uniform(0,1, (self.N, self.IN.shape[0])).astype(self.dtype)
        self.R_ix *= (self.IN[:,1] - self.IN[:,0])
        self.R_ix += self.IN[:,0]

        ### compute cholesky-factorization
        ### this will fail if K is not-PSD LinAlgError: Matrix is not positive definite
        try:
            L_ij = np.linalg.cholesky( self.kernel.K( self.R_ix ) ) ## not immutable
        except:
            #print("not PSD added to diag")
            L_ij = np.linalg.cholesky( self.kernel.K( self.R_ix ) + np.diag( 1.e-8 * np.random.rand(self.N).astype(self.dtype) ) )

        ### compute OUT-space defining-points
        D_iX  = np.random.normal(0,1,(self.N, self.MO)).astype(self.dtype)
        D_iX *= np.diag(L_ij)[:,None]
        D_iX  = np.matmul(L_ij, D_iX)

        self.S_iX  = scipy.linalg.cho_solve((L_ij, True), D_iX)

    def evaluate(self, D_ax):
        """ evaluate for arbitrary values/points in OUT given points in IN.
        GIVEN >
              self
              D_ax : 2d-np.ndarray (D_ax ∈ IN)
        GET   >
              D_aX : 2d-np.ndarray (D_aX ∈ OUT, note captial 'X')
        """
        D_ax += self.IN_noise*np.random.normal(0,1,D_ax.shape).astype(self.dtype)
        D_aX  = np.matmul( self.kernel.K(D_ax, self.R_ix), self.S_iX )
        D_aX += self.OUT_noise*np.random.normal(0,1,D_aX.shape).astype(self.dtype)
        return D_aX

    def get_freq(self):
        return np.linspace(self.d_min, self.d_max, self.d)

    def __call__(self, D_ax):

        df = (self.d_max-self.d_min)/self.d ## :float,  frequnecy resolution of the detector


        D_aX = self.evaluate(D_ax)
        M    = construct_Wigner(D_aX)
        E_ni, v_nij = np.linalg.eigh(M)
        A_nl = v_nij[:,:,0]

        E_ni = ((E_ni-self.d_min)/df).astype(int)
        mask = np.logical_or( (E_ni < 0) , (E_ni > self.d-1) )
        E_ni[mask] = 0
        A_nl[mask] = 0.

        detector_f = np.ones(D_ax.shape[0])[:,None]*np.linspace(self.d_min, self.d_max, self.d)[None,:]  ### detector frequency bins
        detector_A = np.ones(D_ax.shape[0])[:,None]*np.zeros( self.d )[None,:]  ### detector

        ###?
        for n in range(D_ax.shape[0]):
            detector_A[n,E_ni[n,:]] = np.abs(A_nl[n,:])**2 ### IDEAL (stick-figure) detector intensities
        #detector_A[:,E_ni[:]] = np.abs(A_nl[:])**2
        t   = np.arange(self.d) / (self.d_max-self.d_min)
        OUT = np.abs( np.fft.ifft( np.fft.fft( detector_A , axis=1)*np.exp(-self.γ*t)[None,:] , axis=1 ) )
        return OUT