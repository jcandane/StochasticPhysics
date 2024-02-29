import numpy as np
import h5py
import scipy

##### RCF as is : 2/19/2024 #####################

class RCF():
    """
    this an object of a random-contionus function, with-respect-to a GP kernel
    f : IN -> OUT
    can we be given the points? 
    a previous function to compose with? 
    Automatic Derivatives??
    """

    def __init__(self, Domain, X, D=1, kernel=None, ξ=0.1):
        self.domain = Domain ### numpy.2darray
        self.D      = D      ### int (dimension of OUT)
        self.ξ      = ξ
        
        ### get IN points
        if isinstance(X, int): ## if X is int, then get random sampling to define function
            self.D_ix = self.getrandom(X)
        else: ## if X is 2d-np.array, then get uniform grid to define function
            try:
                self.D_ix = self.getgrid(X)
            except:
                raise print("Error")

        μ_i = np.zeros(self.D_ix.shape[0])

        ### cholesky-factor
        Σ_ij      = self.default_kernel(self.D_ix, self.D_ix)
        self.L_ij = np.linalg.cholesky(Σ_ij) ## if using random, it might not be PSD because of point collisions...

        ### calculate y-axis
        Σ_i  = np.diag(Σ_ij)
        D_iX = np.random.normal( μ_i[:,None]*np.ones(self.D)[None,:], Σ_i[:,None]*np.ones(self.D)[None,:], (Σ_i.shape[0],self.D) )
        ## correlate D_iX using the Cholesky-factorization, yielding random/correlated normal-samples
        self.D_iX = self.L_ij @ D_iX ## ~ Y N^2 
        ### careful with the kernel correlation-length this can make things no so smooth!!

    def evalulate(self, D_ax):
        """ evalulate for arbitrary values/points in OUT given points in IN
        GIVEN   : function-values above {D_ix, D_iX, L_ij} : 2d-numpy.array
        GET     : D_aX
        """
        return self.default_kernel(D_ax, self.D_ix) @ scipy.linalg.cho_solve((self.L_ij, True), self.D_iX)

    def getgrid(self, dr_x): ###! spacing: linspace or arange???!!
        """
        get regular grid spacing based on dr_x
        Get: D_ix : numpy.2darray
        """
        R_ix = np.stack(np.meshgrid(*[ np.arange(self.domain[i,0], self.domain[i,1], dr_x[i]) for i in range(len(dr_x)) ]), axis=-1)
        return R_ix.reshape((np.prod( R_ix.shape[:-1] ), R_ix.shape[-1]))

    def getrandom(self, N):
        """ perhaps use Poisson-Disc sampling algorithm to ensure PSD!
        get random points in the domain to define the function
        Get: D_ix : numpy.2darray
        """
        return np.asarray([(element[1]-element[0])*np.random.rand(N) + element[0] for element in self.domain]).T
    
    def default_kernel(self, R_ix, R_jx, ξ=None):
        """
        compute kernel function (RBF) between two domain points

        R_ijx = X_ix - Y_jx
        Σ     = exp( - sum( R_ijx**2 , over=x) / ξ )

        INPUT  : X (X data) : numpy.2darray
                 Y (Y data) : numpy.2darray
                *ξ (correlation length) : float64
        RETURN : Σ : numpy.2darray
        """
        if ξ==None:
            ξ = self.ξ
        R_ij = np.linalg.norm(R_ix[:, None, :] - R_jx[None, :, :], axis=2)
        return np.exp( - R_ij**2 / ξ )

#################################################

def gpcam_to_h5(data, filename="to_vintrumentxx.h5"):
    """ this function reads gpcam's data, and creates an h5 file (to be read by the instrument)
    GIVEN   > data : List[dict] (gpCAM dataset datatype, !contains various datatypes)
            **filename : str (optional, str specifying the output h5 file)
    GET     > None 
    """

    to_analyze=[]
    for entry in data:
        to_analyze.append(entry["x_data"])
    to_analyze = np.asarray(to_analyze) ## make into a np.array, D_ax

    h5f = h5py.File(filename, "w")
    h5f.create_dataset("dataset_1", data=to_analyze)
    h5f.close()
    return None

def h5_to_vinstrument(filename="to_vintrumentxx.h5"):
    """
    this function reads a h5 file, to obtain a 2d-numpy.array (to be used by the virtual-intrument)
    GIVEN > **filename : str
    GET   > x_data : np.ndarray{2d} (D_ax, 1st-index enumerates snapshots, 2nd-index enumerates IN-coordiante, i.e. D_ax) 
    """

    h5f    = h5py.File(filename, "r")
    x_data = np.asarray(h5f.get('dataset_1'))
    h5f.close()
    return x_data ### numpy.array of dimensions ( samples , coordinates ) i.e. D_ax

def vinstrument_to_h5(y_data, filename="from_vintrumentxx.h5"):
    """
    this function obtained the vintrument's y_data, along with other meta-data saves to an h5
    GIVEN > y_data : np.ndarray{2d} (2d-np.array, 1st-index : data-entry number, 2nd-index : OUT-coordinate, i.e. D_aX)
    GET >   None
    """

    h5f = h5py.File(filename, "w")
    h5f.create_dataset("dataset_1", data=y_data)
    h5f.close()
    return None

def h5_to_gpcam(data, filename="from_vintrumentxx.h5"):
    """ this function updates gpcam's "data" variable (List[dict]), by reading a h5 file.
    GIVEN > data : List[dict] (gpCAM dataset datatype, !contains various datatypes)
            **filename : str (optional, str specifying the input h5 file)
    GET   > data : List[dict] (gpCAM dataset datatype, !contains various datatypes)
    """
    h5f    = h5py.File(filename, "r")
    y_data = np.asarray(h5f["dataset_1"]) ## D_aX
    h5f.close()

    for a, entry in enumerate(data):
        entry["y_data"] = np.asarray([y_data[a]]) ### this should have the shape of (2,1) as given in instrument
        entry["output positions"] = np.asarray([np.arange(len(y_data[a]))]).T #np.array([[0],[1]]) ### this is important for fvGP object!
        #entry["output positions"] = np.asarray([np.arange(f.D)]).T

    return data

class gpcam_test_instrument():
    """
    
    """

    def __init__(self, Domain, N, D=1, kernel=None, ξ=0.1):
        self.domain = Domain ### numpy.2darray
        self.D      = D      ### int (dimension of OUT)
        self.ξ      = ξ
        self.N      = N

        self.rcf    =  RCF(self.domain, self.N, D=self.D, ξ=self.ξ)
        

    #################################
    def vinstrument(self):
        """ python-function for virtual-instrument, reads and writes h5 files
        GIVEN > None
        GET   > None
        """

        x_data = h5_to_vinstrument()

        y_data = self.rcf.evalulate(x_data)

        vinstrument_to_h5(y_data)

        return None

    def test_instrument(self, data):

        ### gpcam -> h5 (x-coordinates only)
        gpcam_to_h5(data)

        ### vintrument()
        self.vinstrument()

        ### h5 -> gpcam (everything)
        data = h5_to_gpcam(data)

        return data
    
    def test_instrumentt(self, data):

        ### gpcam -> h5 (x-coordinates only)
        gpcam_to_h5(data)

        ### vintrument()
        self.vinstrument()

        ### h5 -> gpcam (everything)
        data = h5_to_gpcam(data)

        return data
    #################################
