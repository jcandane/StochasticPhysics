import jax
import gpjax as gpx

class RCF():
    """ built: 2/20/2024
    this an object of a Random-Contionus-Function (RCF), with-respect-to a gpJAX kernel
    RCF : IN -> OUT
    we define a prior, and then sample to form a posterior.
    """

    def __init__(self, Domain, N:int, MO:int=1, seed:int=137, kernel=None, μ_i=None):
        """
        Domain : 2d-np.ndarray (domain of input points)
        N : int (number of points)
        MO : int (Multiple-Output Dimension)
        ** seed : int (opinonal, integer to define JAX PRNGKey random-key)
        ** kernel (opinonal, defaults to gpJAX's RBF kernel)
        ** μ_i (opinonal, mean-function, defaults to zeros-Array of length N)
        """
        self.domain = Domain ### numpy.2darray
        self.N      = N      ### number of defining points
        self.MO     = MO     ### int (dimension of OUT)
        if kernel is None:
            self.kernel = gpx.kernels.RBF()
        else:
            try:
                kernel.gram, kernel.cross_covariance ### can we compute a Gram & Cross-Covariance matrix?
                self.kernel = kernel
            except:
                raise "kernel of the wrong type, must be of class gpx.kernels"
        if μ_i is None:
            self.μ_i = jax.numpy.zeros(self.N)
        else:
            self.μ_i = μ_i
        ### define random sampling key
        self.key = jax.random.PRNGKey(seed)
        
        ### find a series of random defining points, keep looping until we find a stable configuration of initial-points
        c_i       = jax.numpy.diff(self.domain, axis=1).reshape(-1)
        self.R_ix = c_i[None,:]*jax.random.uniform(self.key, (N, self.domain.shape[0])) + self.domain[:,0][None,:]

        Σ_ij      = self.kernel.gram(self.R_ix)
        self.L_ij = jax.numpy.linalg.cholesky(Σ_ij.A)
        ###

        Σ_i       = jax.numpy.diag(Σ_ij.A)
        D_iX      = self.μ_i[:,None]*jax.numpy.ones(self.MO)[None,:] + (Σ_i[:,None]*jax.numpy.ones(self.MO)[None,:]) * jax.random.normal( self.key, (self.N,self.MO) ) # Affine-transformation on jax.random.normal
        ## correlate D_iX using the Cholesky-factor, yielding random/correlated normal-samples
        self.D_iX = self.L_ij @ D_iX

    def evalulate(self, D_ax):
        """ evalulate for arbitrary values/points in OUT given points in IN
        GIVEN > self, function-values above {D_ix, D_iX, L_ij} : 2d-numpy.ndarray
        GET   > D_aX : 2d-numpy.ndarray
        """
        return self.kernel.cross_covariance(D_ax, self.R_ix) @ jax.scipy.linalg.cho_solve((self.L_ij, True), self.D_iX)