import numpy as np

class FitBucket(object):
    '''each bucket holds one regression fit plus diagnostics, etc.
    
    Parameters
    ----------
    time_idx : float
        self-identify the time stamp from the Time column of an epochs
        DataFrame
    chan_jdx : uint
        self-identify the channel name from the channel columns of an
        epochs DataFrame

    '''
    def __init__(self,time_idx,chan_jdx):
        self.time_idx=time_idx
        self.chan_jdx=chan_jdx
        
        self.time = None
        self.chan = None
        self.reg_fit = {} # np.ndarray(shape=(0,),dtype=None) # TODO what dtype?
        self.diagnostics = {} 

class FitGrid(np.ndarray):
    """ numpy.ndarray container for FitBuckets: times x channels
 
    Parameters
    ----------
    ntimes : int
        number of time stamp rows in the grid
    nchans : uint
        number of channels columns in the grid
    
    """
    def __new__(cls, ntimes, nchans):
        cls._ntimes = ntimes
        cls._nchans = nchans
        return super(FitGrid, cls).__new__(cls,
                                              shape=(ntimes,nchans),
                                              dtype='object')
    def __init__(self, ntimes, nchans):
        """ initialze with grid of empty FitBuckets """
        for t in range(ntimes):
            for c in range(nchans):
                self[t,c] = FitBucket(t,c)
                
    # generic getter engine #1
    def _get_arry(self, field, dtype):
        """generic array getter for scalar field of dtype


        Parameters
        ---------

        field : string
          name of a FitBucket attribute to extract

        dtype : numpy-friendly data type

        Returns:
          arry : ndarray of field values where arry.dtype == dtype

        Notes:

        * str is not a numpy-friendly dtype, use object for strings

        * calling this on slices is blocked for now b.c. ndarray[:,0]
          and ndarray[0,:] both have shape==(n,) so 1-D slices are
          row-column ambiguous

        """
        # TO DO: allow slices if 1-D dims == self._ntimes or ._chans

        # block if self is a slice of the 
        if self.shape != (self._ntimes,self._nchans):
            msg = ( 'FitGrid[:,:].field slices not supported, try using'
                    'FitGrid.field[:,:]' )
            raise RuntimeError(msg)
        arry = np.ndarray(self.shape, dtype=dtype)
        for t in range(self.shape[0]):
            for c in range(self.shape[1]):
                arry[t,c] = getattr(self[t,c], field)
        return(arry)
    
    # public getters for FitBucket attributes, add more as needed
    @property
    def time(self):
        """returns the ndarray of times"""
        return self._get_arry('time', int)
   
    @property
    def chan(self):
        """return ndarray of channel names"""
        return self._get_arry('chan', np.dtype(object))
                  
    @property
    def reg_fit(self):
        """return regression fits dtype = object)""" 
        return self._get_arry('reg_fit', np.dtype(object))
    
    @property
    def diagnostics(self):
        """return diagnostics (dtype = object)"""
        return self._get_arry('diagnostics', np.dtype(object))


