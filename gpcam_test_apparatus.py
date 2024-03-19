import numpy as np
import h5py

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

class gpcam_test_apparatus():
    """
    
    """

    def __init__(self, Domain, N:int, MO:int=1, seed:int=777,
                 IN_noise=None, OUT_noise=None,
                 kernel=None):
      
        if isinstance(Domain, np.ndarray):
            from rcf_gpy import RCF
        
        elif str( type( Domain ) ).split(" ")[1].split("'")[1].split(".")[0] == "jaxlib":
            from rcf_jax import RCF

        elif str( type( Domain ) ).split(" ")[1].split("'")[1].split(".")[0] == "torch":
            from rcf_torch import RCF
        
        elif str( type( Domain ) ).split(" ")[1].split("'")[1].split(".")[0] == "tensorflow":
            from rcf_tf import RCF

        if kernel is None:
            self.rcf=RCF(Domain, N, MO, seed, IN_noise, OUT_noise)
        else:
            self.rcf=RCF(Domain, N, MO, seed, IN_noise, OUT_noise, kernel=kernel)
        

    #################################
    def vinstrument(self):
        """ python-function for virtual-instrument, reads and writes h5 files
        GIVEN > None
        GET   > None
        """

        x_data = h5_to_vinstrument()

        y_data = self.rcf.evaluate(x_data)

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