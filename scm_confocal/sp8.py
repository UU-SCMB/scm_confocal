import glob
import numpy as np
import os

class sp8_series:
    """
    Class of functions related to the sp8 microscope. The functions assume that
    the data are exported as .tif files and placed in a own folder per series.
    The current working directory is assumed to be that folder. For several
    functions it is required that the xml metadata is present in a subfolder of
    the working directory called 'MetaData', which is normally generated
    automatically when exporting tif files as raw data.
    
    Attributes
    ----------
    filenames : list of str
        the filenames loaded associated with the series
    data : numpy array
        the image data as loaded on the most recent call of sp8_series.load_data()
    metadata : xml.Elementtree root
        the recording parameters associated with the image series
    """
    
    def __init__(self,fmt='*.tif'):
        """
        Initialize the class instance and assign the filenames of the data.

        Parameters
        ----------
        fmt : str, optional
            format to use for finding the files. Uses the notation of the glob
            library. The default is '*.tif'.


        Returns
        -------
        None.

        """

        self.filenames = glob.glob(fmt)
        if len(self.filenames) < 1:
            raise ValueError('No images found in current directory')
        
    def load_data(self, filenames=None, first=None, last=None, dtype=np.uint8):
        """
        Loads the sequence of images into ndarray of form (files,y,x) and
        converts the data to dtype

        Parameters
        ----------
        filenames : list of str, optional
            filenames of images to load. The default is what is passed from 
            __init__, which by default is all .tif images in the current
            working directory.
        first : None or int, optional
            index of first image to load. The default is None.
        last : None or int, optional
            index of last image to load plus one. The default is None.
        dtype : (numpy) datatype, optional
            type to scale data to. The default is np.uint8.

        Returns
        -------
        data : numpy.ndarray
            3d numpy array with dimension order (filenames,y,x).

        """
        import PIL.Image
        
        if filenames == None:
            filenames = sorted(self.filenames)
        
        data = np.array([np.array(PIL.Image.open(name)) for name in filenames[first:last]])

        #check if images are 2D (i.e. greyscale)
        if data.ndim > 3:
            print("[WARNING] sp8_series.load_data(): images do not have the correct dimensionality, "+
                  "did you load colour images perhaps? Continueing with average values of higher dimensions")
            data = np.mean(data,axis=tuple(range(3,data.ndim)),dtype=dtype)


        if not data.dtype == dtype:
            data = data/np.amax(data)*np.iinfo(dtype).max
            data = data.astype(dtype)
        self.data = data
        return data
    
    def load_stack(self,dim_range={},dtype=np.uint8):
        """
        Similar to sp8_series.load_data(), but converts the 3D array of images
        automatically to a np.ndarray of the appropriate dimensionality.
        
        Array dimensions are specified as follows:
        
            - If the number of detector channels is 2 or higher, the first
              array axis is the detector channel index (named 'channel').
              
            - If the number of channels is 1, the first array axis is the first
              available dimension (instead of 'channel').
              
            - Each subsequent array axis corresponds to a dimension as
              specified by and in reversed order of the metadata exported by
              the microscope software, excluding dimensions which are not
              available. The default order of dimensions in the metadata is:
              
                 - (0 = 'channel')
                 -  1 = 'x-axis'
                 -  2 = 'y-axis'
                 -  3 = 'z-axis'
                 -  4 = 'time'
                 -  5 = 'detection wavelength'
                 -  6 = 'excitation wavelength'
                 
            - As an example, a 2 channel xyt measurement would result in a 4-d
              array with axis order ('channel','time','y-axis',
              'x-axis'), and a single channel xyz scan would be returned as
              ('z-axis','y-axis','x-axis')
        
        For loading only part of the total dataset, the dim_range parameter can
        be used to specify a range along any of the dimensions. This will be
        more memory efficient than loading the entire stack and then discarding
        part of the data. For slicing along the x or y axis this is not
        possible and whole (xy) images must be loaded prior to discarding
        data outside the specified x or y axis range.

        Parameters
        ----------
        dim_range : dict, optional
            dict, with keys corresponding to channel/dimension labels as above
            and slice objects as values. This allows you to only load part of
            the data along any of the dimensions, such as only loading one
            channel of multichannel data or a particular z-range. An example
            use for only taking time steps up to 5 and z-slice 20 to 30 would
            be:
            
                dim_range={'time':slice(None,5), 'z-axis':slice(20,30)}.
                
            The default is {}.
        dtype : (numpy) datatype, optional
            type to scale data to. The default is np.uint8.

        Returns
        -------
        data : numpy.ndarray
            ndarray with the pixel values
        dimorder : tuple
            tuple with lenght data.ndim specifying the ordering of dimensions
            in the data with labels from the metadata of the microscope.
        """

        #load the metadata
        try:
            channels = self.metadata_channels
        except AttributeError:
            channels = sp8_series.get_metadata_channels(self)
        try:
            dimensions = self.metadata_dimensions
        except AttributeError:
            dimensions = sp8_series.get_metadata_dimensions(self)
        
        #determine what the new shape should be from dimensional metadata
        newshape = [int(dim['NumberOfElements']) for dim in reversed(dimensions)]
        
        #create list of dimension labels
        order = [sp8_series._DimIDreplace(dim['DimID']) for dim in reversed(dimensions)]
        
        #append channel (but before x and y) information for multichannel data
        if len(channels)>1:
            newshape = newshape[:-2] + [len(channels)] + newshape[-2:]
            order = order[:-2] + ['channel'] + order[-2:]
        
        #load filenames
        filenames = self.filenames

        #apply slicing to the list of filenames before loading images
        if len(dim_range) > 0:

            self._stack_dim_range = dim_range

            #give a warning that only whole xy images are loaded
            if 'x-axis' in dim_range or 'y-axis' in dim_range:
                print("[WARNING] confocal.sp8_series.load_stack: Loading only"+
                      " part of the data along dimensions 'x-axis' and/or "+
                      "'y-axis' not implemented. Data will be loaded fully "+
                      "into memory before discarding values outside of the "+
                      "slice range specified for the x-axis and/or y-axis. "+
                      "Other axes for which a range is specified will still "+
                      "be treated normally, avoiding unneccesary memory use.")
            
            #give warning for nonexistent dimensions
            if len(dim_range.keys() - set(order)) > 0:
                for dim in dim_range.keys() - set(order):
                    print("[WARNING] confocal.sp8_series.load_stack: "+
                          "dimension '"+dim+"' not present in data, ignoring "+
                          "this entry.")
                    dim_range.pop(dim)
            
            #create a tuple of slice objects for each dimension except x and y
            slices = []
            for dim in order[:-2]:
                if not dim in dim_range:
                    dim_range[dim] = slice(None,None)
                slices.append(dim_range[dim])
            slices = tuple(slices)
            
            #reshape the filenames and apply slicing, then ravel back to flat list
            filenames = np.reshape(filenames,newshape[:-2])[slices]
        
        else:
            filenames = np.reshape(filenames,newshape[:-2])
            
        #change dim order if multiple channels, move to 0th axis
        for i,dim in enumerate(order):
            if dim == 'channel':
                filenames = np.moveaxis(filenames,i,0)
                order = [order[i]] + order[:i] + order[i+1:]
        
        #get final shape and flatten list of filenames in correct order
        newshape = list(np.shape(filenames)) + newshape[-2:]
        filenames = list(filenames.ravel())
        
        #load and reshape data
        data = self.load_data(filenames=filenames,dtype=dtype)
        data = np.reshape(data,tuple(newshape))
        
        #if ranges for x or y are chosen, remove those from the array now,
        #account (top to bottom) for trimming x ánd y, only x, or only y.
        if 'x-axis' in dim_range:
            if 'y-axis' in dim_range:
                slices = tuple([slice(None)]*len(newshape[:-2]) + [dim_range['y-axis'],dim_range['x-axis']])
            else:
                slices = tuple([slice(None)]*len(newshape[:-1]) + [dim_range['x-axis']])
            data = data[slices]
        elif 'y-axis' in dim_range:
            slices = tuple([slice(None)]*len(newshape[:-2]) + [dim_range['y-axis']])
            data = data[slices]
        
        return data, tuple(order)
    
    def load_metadata(self):
        """
        Load the xml metadata exported with the files as xml_root object which
        can be indexed with xml.etree.ElementTree

        Returns
        -------
        metadata : xml.etree.ElementTree object
            Parsable xml tree object containing all the metadata

        """
        import xml.etree.ElementTree as et
        
        metadata_path = sorted(glob.glob(os.path.join(os.path.curdir, 'MetaData', '*.xml')))[0]
        metadata = et.parse(metadata_path)
        metadata = metadata.getroot()
        
        self.metadata = metadata
        return metadata
    
    def get_metadata_channels(self):
        """
        Gets the channel information from the metadata

        Returns
        -------
        channels : list of dict
            list of dictionaries with length equal to number of channels where
            each dict contains the metadata for one channel
        """
        #Fetch metadata from class instance or load if it was not loaded yet
        try:
            metadata = self.metadata
        except AttributeError:
            metadata = sp8_series.load_metadata(self)
        
        channels = [dict(ch.attrib) for ch in metadata.find('.//Channels')]
        
        self.metadata_channels = channels
        return channels
    
    def get_metadata_dimensions(self):
        """
        Gets the dimension information from the metadata

        Returns
        -------
        dimensions : list of dict
            list of dictionaries with length number of dimensions where
            each dict contains the metadata for one data dimension

        """
        #Fetch metadata from class instance or load if it was not loaded yet
        try:
            metadata = self.metadata
        except AttributeError:
            metadata = sp8_series.load_metadata(self)
        
        dimensions = [dict(dim.attrib) for dim in metadata.find('.//Dimensions')]
        
        self.metadata_dimensions = dimensions
        return dimensions
    
    def get_metadata_dimension(self,dim):
        """
        Gets the dimension data for a particular dimension. Dimension can be
        given both as integer index (as specified by the Leica exported 
        MetaData which may not correspond to the indexing order of the data
        stack) or as string containing the physical meaning, e.g. 'x-axis',
        'time', 'excitation wavelength', etc.

        Parameters
        ----------
        dim : int or str
            dimension to get metadata of specified as integer or as name.

        Returns
        -------
        dimension : dict
            dictionary containing all metadata for that dimension

        """
        #convert string labels to corresponding integer labels
        if dim == 'channel' or dim == 0:
            raise ValueError('use sp8_series.get_metadata_channels() for '+
                                      'channel data')
        elif dim == 'x-axis':
            dim = 1
        elif dim == 'y-axis':
            dim = 2
        elif dim == 'z-axis':
            dim = 3
        elif dim == 'time':
            dim = 4
        elif dim == 'detection wavelength':
            dim = 5
        elif dim == 'excitation wavelength':
            dim = 6
        elif type(dim) != int or dim>6 or dim<0:
            raise ValueError('"'+str(dim)+'" is not a valid dimension label')
        
        #fetch or load dimensions
        try:
            dimensions = self.metadata_dimensions
        except AttributeError:
            dimensions = self.get_metadata_dimensions()
        
        #find correct dimension in the list of dimensions
        index = [int(d['DimID']) for d in dimensions].index(dim)
        dimension = dict(dimensions[index])
        
        return dimension
    
    def get_dimension_steps(self,dim,load_stack_indices=False):
        """
        Gets a list of values for each step along the specified dimension, e.g.
        a list of timestamps for the images or a list of height values for all
        slices of a z-stack. For specification of dimensions, see
        sp8_series.get_metadata_dimension()

        Parameters
        ----------
        dim : int or str
            dimension to get steps for
        load_stack_indices : bool
            if True, trims down the list of steps based on the dim_range used
            when last loading data with load_stack

        Returns
        -------
        steps : list
            list of values for every logical step in the data
        unit : str
            physical unit of the step values

        """
        #get the data
        dimension = self.get_metadata_dimension(dim)
        
        #obtain the infomation and calculate steps
        start = float(dimension['Origin'])
        size = float(dimension['Length'])
        n = int(dimension['NumberOfElements'])
        unit = dimension['Unit']
        steps = np.linspace(start,start+size,n)

        if load_stack_indices:
            try:
                dim_range = self._stack_dim_range
            except AttributeError:
                raise AttributeError('data must be loaded with '+
                                  'sp8_series.load_stack() prior to '+
                                  'calling visitech_faststack.get_timestamps()'
                                  +' with load_stack_indices=True')

            if dim in dim_range:
                steps = steps[dim_range[dim]]

        return steps,unit
    
    def get_dimension_stepsize(self,dim):
        """
        Get the size of a single step along the specified dimension, e.g.
        the pixelsize in x, y or z, or the time between timesteps. For
        specification of dimensions, see sp8_series.get_metadata_dimension()

        Parameters
        ----------
        dim : int or str
            dimension to get stepsize for

        Returns
        -------
        value : float
            stepsize
        unit : int
            physical unit of value

        """
        #get the data
        dimension = self.get_metadata_dimension(dim)
        
        #obtain the infomation and calculate steps
        size = float(dimension['Length'])
        n = int(dimension['NumberOfElements'])
        unit = dimension['Unit']
        
        return size/(n-1),unit
    
    def get_series_name(self):
        """
        Returns a string containing the filename (sans file extension) under 
        which the series is saved.

        Returns
        -------
        name : str
            name of the series

        """

        #find metadata file in subfolder, split off location and extension
        path = os.path.join(os.path.curdir, 'MetaData', '*.xml')
        path = sorted(glob.glob(path))[0]
        return os.path.split(path)[1][:-4]
    
    def _DimIDreplace(idlist):
        """replaces dimID int with more sensible label"""
        pattern = zip(list('123456'),['x-axis','y-axis','z-axis','time',
                      'detection wavelength','emission wavelength'])
        for l,r in pattern:
            idlist = idlist.replace(l,r)
        return idlist
