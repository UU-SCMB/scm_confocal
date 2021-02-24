import glob
import numpy as np
import os

class sp8_lif:
    """
    Class of functions related to the sp8 microscope, for data saves as .lif 
    files, the default file format for the Leica LAS-X software. Essentially
    a wrapper around the `readlif` library, which provides access to the data 
    and metadata directly in Python.
    
    The underlying `readlif.LifFile` instance can be accessed directly using 
    the `sp8_lif.liffile` attribute, and any of it attributes are accessible 
    through `sp8_lif` directly.
    
    Parameters
    ----------
    filename : str
        Filename of the `.lif` file. Extension may be (but is not required to 
        be) included.
    quiet : bool, optional
        can be used to suppress printing the contents of the file. The default
        is False.

    Returns
    -------
    `sp8_lif` class instance
    
    Attributes
    ----------
    liffile : readlif.LifFile instance
        The underlying class instance of the readlif library.
    filename : str
        filename of the loaded .lif file with file extention included, even if
        it was not given when initializing the class.
    
    See also
    --------
    sp8_image(), a subclass for specific images in the dataset.
    
    [readlif](https://github.com/nimne/readlif), the library used for 
    acessing the files.
        
    """
    def __init__(self,filename,quiet=False):
        """
        Initialize the class instance and the underlying LifFile instance
        """
        from readlif.reader import LifFile
        
        #try reading, if fails try again with extension appended
        try:
            self.liffile = LifFile(filename)
            self.filename = filename
        except FileNotFoundError:
            try:
                self.liffile = LifFile(filename+'.lif')
                self.filename = filename+'.lif'
            except FileNotFoundError:
                raise FileNotFoundError("No such file or directory: '"+str(filename)+"'")
        
        #for convenience print contents of file
        if not quiet:
            self.print_images()

    
    def __getattr__(self,attrName):
        """
        Automatically called when getattribute fails. Delegate parent attribs
        from LifFile
        """
        try:
            return getattr(self.liffile,attrName)
        except AttributeError:
            raise AttributeError('sp8_lif object has no attribute %s' % attrName)
            
    def print_images(self):
        """
        for convenience print basic info of the datasets in the lif file
        
        the format is <image index>: <number of channels>, <dimensions>
        """
        for i,im in enumerate(self.image_list):
            print('{:}: {:}, {:} channels, {:}'.format(i,im['name'],im['channels'],im['dims']))
    
    def get_image(self,image=0):
        """
        returns an sp8_image instance containing relevant attributes and 
        functions for the specific image in the dataset, which provides the 
        "bread and butter" of data access.

        Parameters
        ----------
        image : int or str, optional
            The image (or image series) to obtain. May be given as index number
            (int) or as the name of the series (string). The default is the 
            first image in the file.

        Returns
        -------
        `sp8_image` class instance

        """
        return sp8_image(self.filename,self._image_name_to_int(image))
    
    def get_liffile_image(self,image=0):
        """
        returns the `readlif.LifImage` instance for a particular image in the
        dataset.

        Parameters
        ----------
        image : int or str, optional
            The image (or image series) to obtain. May be given as index number
            (int) or as the name of the series (string). The default is the 
            first image in the file.

        Returns
        -------
        `readlif.LifImage` class instance
        """
        return self.liffile.get_image(self._image_name_to_int(image))
    
    def _image_name_to_int(self,image):
        """shortcut for converting image name to integer for accessing data"""
        
        #check input
        if not isinstance(image,(str,int)):
            raise TypeError('`image` must be of type `int` or `str`')
        
        #if string, convert to int
        elif isinstance(image,str):
            try:
                image = [im['name'] for im in self.image_list].index(image)
            except ValueError:
                raise ValueError('{:} it not in {:}'.format(image,self.filename))
                
        return image
    
    def _image_int_to_name(self,image):
        """shortcut for converting image name to integer for accessing data"""
        
        #check input
        if not isinstance(image,(str,int)):
            raise TypeError('`image` must be of type `int` or `str`')
        
        #if string, convert to int
        elif isinstance(image,int):
            if image >= self.num_images:
                raise ValueError(str(image)+' not in list of images')
            image = self.image_list[image]['name']
                
        return image
    
class sp8_image(sp8_lif):
    """
    Subclass of `sp8_lif` for relevant attributes and functions for a specific
    image in the .lif file. Should not be called directly, but rather be 
    obtained through `sp8_lif.get_image()`
    
    Parameters
    ----------
    filename : str
        file name of the parent .lif file
    image : int
        index number of the image in the parent .lif file
    
    Attributes
    ----------
    image : int
        index number of the image in the parent .lif file
    lifimage : `readlif.LifImage` class instance
        The underlying class instance of the readlif library.
    
    Additionally, attributes and functions of the parent `sp8_lif` instance are
    inherited and directly accessible, as well as all attributes of the 
    `readlif.LifImage` instance.
    """
    def __init__(self,filename,image):
        """inherit all functions and attributes from parent sp8_lif class and 
        add some image specific ones"""
        super().__init__(filename,quiet=True)
        self.image = image
        self.lifimage = self.liffile.get_image(self.image)
    
    def __getattr__(self,attrName):
        """
        Automatically called when getattribute fails. Delegate parent attribs
        from LifFile
        """
        try:
            return getattr(self.lifimage,attrName)
        except AttributeError:
            raise AttributeError('sp8_lif object has no attribute %s' % attrName)
    
    def get_name(self):
        """
        shortcut for getting the name of the dataset / image for e.g. 
        automatically generating filenames for stored results.
        
        The format is: <lif file name (without file extension)>_<image name>
        """
        return self.filename.rpartition('.')[0]+'_'+self.name
      
    def get_metadata(self):
        """
        parse the .lif xml data for the current image
        
        Returns
        -------
        `xml.etree.ElementTree` instance for the current image
        """
        try:
            self.metadata
        except AttributeError:
            self.metadata = \
                self.liffile.xml_root.find('.//Children').findall('Element')[self.image]
        return self.metadata
        
    def get_channels(self):
        """
        parse the images xml data for the channels.
        
        Returns
        -------
        list of dictionaries
        """
        try:
            return self.metadata_channels
        except AttributeError:    
            root = self.get_metadata()
            self.metadata_channels = [dict(dim.attrib) for dim in root.find('.//Channels')]
        return self.metadata_channels
    
    def get_channel(self,chl):   
        """
        get info from the metadata on a specific channel

        Parameters
        ----------
        chl : int
            index number of the channel.

        Returns
        -------
        channel: dict
            dictionary containing all metadata for that channel
        """
        #check input
        if not isinstance(chl,int):
            raise TypeError('`chl` must be of type `int`')
        if chl >= self.channels:
            raise ValueError('channel '+str(chl)+' not present in image')
        return dict(self.get_channels()[chl])
    
    def get_dimensions(self):
        """
        parse the images xml data for the dimensions.
        
        Returns
        -------
        list of dictionaries
        """
        try:
            return self.metadata_dimensions
        except AttributeError:    
            root = self.get_metadata()
            self.metadata_dimensions = [dict(dim.attrib) for dim in root.find('.//Dimensions')]
        return self.metadata_dimensions
    
    def get_dimension(self,dim):
        """
        Gets the dimension data for a particular dimension of an image. 
        Dimension can be given both as integer index (as specified by the Leica
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
        if isinstance(dim,str):
            dim = _DimID_to_int(dim)
        
        #check inputs
        if not isinstance(dim,int):
            raise TypeError('dim must be of type `str` or `int`')
        elif dim == 0:
            raise ValueError('use `get_channels()` for channel data')
        elif dim in (6,7,8) or dim>10:
            raise ValueError('"'+str(dim)+'" is not a known dimension')
            
        #load dimensions
        dims = self.get_dimensions()
        
        #find correct dimension in the list of dimensions
        index = [int(d['DimID']) for d in dims].index(dim)
        
        return dict(dims[index])
    
    def get_dimension_stepsize(self,dim):
        """
        returns the step size along a dimension, e.g. time interval, pixel
        size, etc, as (value, unit) tuple. Dimension can be given both as 
        integer index (as specified by the Leica MetaData, which may not 
        correspond to the indexing order of the data stack), or as string 
        containing the physical meaning, e.g. 'x-axis', 'time', 'excitation 
        wavelength', etc.

        Parameters
        ----------
        dim : int or str
            dimension to get metadata of specified as integer or as name.

        Returns
        -------
        stepsize : float
            physical size of one step (e.g. pixel, time interval, ...).
        unit: str
            physical unit of the data.

        """
        dim = self.get_dimension(dim)
        if int(dim['NumberOfElements'])==1:
            stepsize = float(dim['Length'])/int(dim['NumberOfElements'])
        else:
            stepsize = float(dim['Length'])/(int(dim['NumberOfElements'])-1)
        return (stepsize,dim['Unit'])
    
    def get_dimension_steps(self,dim):
        """
        returns a list of corresponding physical values for all steps along
        a given dimension, e.g. a list of time steps or x coordinates.
        Dimension can be given both as integer index (as specified by the Leica
        MetaData, which may not correspond to the indexing order of the data 
        stack), or as string containing the physical meaning, e.g. 'x-axis', 
        'time', 'excitation wavelength', etc.

        Parameters
        ----------
        dim : int or str
            dimension to get metadata of specified as integer or as name.

        Returns
        -------
        steps: list of float
            physical values of the steps along the chosen dimension, (e.g. 
            a list of pixel x-coordinates, list of time stamps, ...).
        unit: str
            physical unit of the data.

        """
        dim = self.get_dimension(dim)
        start = float(dim['Origin'])
        length = float(dim['Length'])
        steps = int(dim['NumberOfElements'])
        return np.linspace(start,start+length,steps),dim['Unit']
        
    def get_pixelsize(self):
        """
        shorthand for `get_dimension_stepsize()` to get the pixel/voxel size
        converted to nanometer, along whatever spatial dimensions are present 
        in the data. Is given as (z,y,x) where dimensions not present in the 
        data are skipped.
        
        Returns
        -------
        pixelsize : tuple of float
            physical size in nm of the pixels/voxels along (z,y,x)
        """
        
        pixelsize = []
        for d in ['z-axis','y-axis','x-axis']:
            try:
                pixelsize.append(self.get_dimension_stepsize(d)[0]*1e9)
            except ValueError:
                pass
        return tuple(pixelsize)
    
    def load_stack(self,dim_range={}):
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
            -  0 = 'channel'
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
        #determine get the varied dimensions
        #dimensions = self.get_dimensions()
        #order = [_DimID_to_str(dim['DimID']) for dim in reversed(dimensions)]
        #order = ['channel'] + order
        order = ['channel','time','z-axis','y-axis','x-axis']
        
        #store slicing
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
        
        #create a tuple with a slice objects for each dimension except x and y
        for dim in order[:-2]:
            if not dim in dim_range:
                dim_range[dim] = slice(None,None)
        
        #set x and y sizes
        try:
            nx = int(self.get_dimension(1)['NumberOfElements'])
        except ValueError:
            nx = 1
        try:
            ny = int(self.get_dimension(2)['NumberOfElements'])
        except ValueError:
            ny = 1
        
        #create list of indices for each dimension and slice them
        channels = np.array(range(self.channels))[dim_range['channel']]
        times = np.array(range(self.nt))[dim_range['time']]
        zsteps = np.array(range(self.nz))[dim_range['z-axis']]
        
        #determine shape and init array
        newshape = (len(channels),len(times),len(zsteps),ny,nx)
        data = np.empty(newshape,dtype=np.uint8)
        
        #loop over indices and load
        for i,c in enumerate(channels):
            for j,t in enumerate(times):
                for k,z in enumerate(zsteps):
                    data[i,j,k,:,:] = self.lifimage.get_frame(z,t,c)
        
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
        
        #squeeze out dimensions with only one element
        dim_order = []
        for i,s in enumerate(data.shape):
            if s > 1:
                dim_order.append(order[i])
        data = np.squeeze(data)

        return data, tuple(dim_order)
     
        
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
        #sort filenames
        if filenames == None:
            filenames = sorted(self.filenames)

        #this ugly try-except block tries different importers for availability
        try:
            #check pillow import
            from PIL.Image import open as imopen
            data = np.array([np.array(imopen(name)) for name in filenames[first:last]])
            data[0]*1
        except:
            print('[WARNING] scm_confocal.load_data: could not import with PIL'+
                  ', retrying with scikit-image. Make sure libtiff version >= '+
                  '4.0.10 is installed')
            try:
                from skimage.io import imread
                data = np.array([imread(name) for name in filenames[first:last]])
            except:
                raise ImportError('could not load data, check if pillow/PIL or '+
                                  ' scikit-image are installed')

        #check if images are 2D (i.e. greyscale)
        if data.ndim > 3:
            print("[WARNING] sp8_series.load_data(): images do not have the correct dimensionality, "+
                  "did you load colour images perhaps? Continueing with average values of higher dimensions")
            data = np.mean(data,axis=tuple(range(3,data.ndim)),dtype=dtype)

        #optionally fix dtype of data
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
                 -  9 = 'excitation wavelength'
                 
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
        order = [_DimID_to_str(dim['DimID']) for dim in reversed(dimensions)]
        
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
        if isinstance(dim,str):
            dim = _DimID_to_int(dim)
        
        #check inputs
        if not isinstance(dim,int) or dim == 0:
            raise ValueError('use sp8_series.get_metadata_channels() for '+
                                      'channel data')
        elif dim > 6:
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
    
def _DimID_to_str(dim):
    """replaces a dimID integer with more sensible string labels"""
    #names and labels
    names = ['channel','x-axis','y-axis','z-axis','time',
             'detection wavelength','excitation wavelength','mosaic']
    labels = [0,1,2,3,4,5,9,10]
    
    #check input
    if isinstance(dim,str):
        dim = int(dim)
    if dim not in labels:
        raise ValueError(str(dim)+' is not a valid dimension label')
    
    return names[labels.index(dim)]

def _DimID_to_int(dim):
    """replaces a dimID string with corresponding integer"""
    #names and labels
    names = ['channel','x-axis','y-axis','z-axis','time',
             'detection wavelength','excitation wavelength','mosaic']
    labels = [0,1,2,3,4,5,9,10]
    
    #check input
    if dim not in names:
        raise ValueError(str(dim)+' is not a valid dimension label')
    
    return labels[names.index(dim)]