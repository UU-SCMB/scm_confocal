# -*- coding: utf-8 -*-
import glob
import numpy as np
import os

class sp8_lif:
    """
    Class of functions related to the sp8 microscope, for data saved as .lif 
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
    sp8_image : a subclass for specific images in the dataset.
    readlif:
        the library used for acessing the files, which can be found 
        [here](https://github.com/nimne/readlif)
        
    """
    def __init__(self,filename=None,quiet=False):
        """
        Initialize the class instance and the underlying LifFile instance
        """
        from readlif.reader import LifFile
        
        #try reading, if fails try again with extension appended
        try:
            if filename is None:
                filename = glob.glob('*.lif')[0]
            
            self.liffile = LifFile(filename)
            self.filename = filename
        except FileNotFoundError:
            try:
                self.liffile = LifFile(filename+'.lif')
                self.filename = filename+'.lif'
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No such file or directory: '{filename}'"
                )
        #for convenience print contents of file
        if not quiet:
            print(self)

    def __getattr__(self,attrName):
        """Automatically called when getattribute fails. Delegate parent 
        attribs from LifFile"""
        if hasattr(self.liffile,attrName):
            return getattr(self.liffile,attrName)
        else:
            raise AttributeError(self,'has no attribute',attrName)
    
    def __len__(self):
        """"define len(self) as number of datasets/images"""
        return len(self.image_list)
    
    def __getitem__(self,i):
        """allows using indexing as a shorthand for `get_image()`"""
        if i >= len(self):
            raise IndexError(f"Index {i} is out of range for image_list with"\
                              f" size {len(self)}")
        return self.get_image(i)
    
    def __repr__(self):
        """returns string representing the object in the interpreter"""
        return f"scm_confocal.sp8_lif('{self.filename}')"
    
    def __str__(self):
        """for convenience print basic info of the datasets in the lif file
        the format is `<image index>: <number of channels>, <dimensions>`"""
        s = f"<scm_confocal.sp8_lif('{self.filename}')>\n"
        for i,im in enumerate(self.image_list):
            s+=f"{i}: {im['name']}, {im['channels']} channels, {im['dims']}\n"
        return s[:-1]#strips last newline
    
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
                raise ValueError(f'{image} it not in {self.filename}')
                
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
    lifimage : readlif.LifImage class instance
        The underlying class instance of the readlif library.
    
    Additionally, attributes and functions of the parent `sp8_lif` instance are
    inherited and directly accessible, as well as all attributes of the 
    `readlif.LifImage` instance.
    """
    def __init__(self,filename,image):
        """inherit all functions and attributes from parent sp8_lif class and 
        add some image specific ones"""
        
        #inherit parent attribs and initialize readlif.LifImage class
        super().__init__(filename,quiet=True)
        self.image = image
        self.lifimage = self.liffile.get_image(self.image)
        
        #note if it is single or multichannel
        if self.lifimage.channels > 1:
            self._is_multichannel = True
        else:
            self._is_multichannel = False
    
    def __getattr__(self,attrName):
        """Automatically called when getattribute fails. Delegate parent 
        attribs from LifFile"""
        if hasattr(self.lifimage,attrName):
            return getattr(self.lifimage,attrName)
        else:
            raise AttributeError(self,'has no attribute',attrName)

    def __len__(self):
        """length of image (series), given as number of images where an image
        is defined by the first two dimensions in recording order, where all
        channels are considered as part of the same frame"""
        #only calculate length once and store as _len attribute
        if hasattr(self,'_len'):
            return self._len
        else:
            dims = self.get_dimensions()
            if len(dims) <= 2:
                self._len = 1
            else:
                self._len = np.product([int(d['NumberOfElements']) \
                                        for d in dims[2:]])
            return self._len

    def __getitem__(self,key):
        """make indexable, where it returns the ith frame, where a frame is 
        defined by the first 2 dimensions in recording order"""
        if isinstance(key,slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        else:
            return self.load_frame(key)
    
    def __iter__(self):
        """initialize iterator"""
        self._iter_n = 0
        return self

    def __next__(self):
        "make iterable where it returns one image at a time"
        self._iter_n += 1
        if self._iter_n >= len(self):
            raise StopIteration
        else:
            return self[self._iter_n]
            
    def __repr__(self):
        """returns string representing the object in the interpreter"""
        return f"scm_confocal.sp8_image('{self.filename}',{self.image})"
    
    def __str__(self):
        """for convenience print basic info about the image"""
        return f"<scm_confocal.sp8_image('{self.filename}',{self.image})>\n" +\
            f'file:\t\t{self.filename}\n' +\
            f'image:\t\t{self.name}\n' +\
            f'channels:\t{self.channels}\n' +\
            f'shape:\t\t{self.dims}'

    def get_name(self):
        """
        shortcut for getting the name of the dataset / image for e.g. 
        automatically generating filenames for stored results.
        
        The format is: `<lif file name (without file extension)>_<image name>`
        """
        return self.filename.rpartition('.')[0]+'_'+self.name
      
    def get_metadata(self):
        """
        parse the .lif xml data for the current image
        
        Returns
        -------
        `xml.etree.ElementTree` instance for the current image
        """
        if hasattr(self,'metadata'):
            return self.metadata
        else:
            self.metadata = \
                self.liffile.xml_root.find('.//Children'
                                           ).findall('Element')[self.image]
            return self.metadata
        
    def get_channels(self):
        """
        parse the images xml data for the channels.
        
        Returns
        -------
        list of dictionaries
        """
        if hasattr(self,'metadata_channels'):
            return self.metadata_channels
        else:   
            root = self.get_metadata()
            self.metadata_channels = [dict(dim.attrib) \
                                      for dim in root.find('.//Channels')]
        return self.metadata_channels
    
    def get_detector_settings(self):
        """
        Parses the xml metadata for the detector settings.
        
        Returns
        -------
        dictionary or (in case of multichannel data) a list thereof
        """
        #get detector data
        detectors = self.get_metadata().findall('.//Detector')
        detectors = [d.attrib for d in detectors]
        
        #select only active detectors
        detectors = [d for d in detectors if d['IsActive']=='1']
        
        #return list for multichannel or first active detector for single
        #channel data
        if self._is_multichannel:
            return detectors
        else:
            return detectors[0]
        
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
        if hasattr(self,'metadata_dimensions'):
            return self.metadata_dimensions
        else:    
            root = self.get_metadata()
            self.metadata_dimensions = [dict(dim.attrib) \
                                        for dim in root.find('.//Dimensions')]
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
    
    def get_dimension_steps(self,dim,load_stack_indices=False):
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
        dimdata = self.get_dimension(dim)
        start = float(dimdata['Origin'])
        length = float(dimdata['Length'])
        nsteps = int(dimdata['NumberOfElements'])
        steps = np.linspace(start,start+length,nsteps)
        
        if load_stack_indices:
            try:
                dim_range = self._stack_dim_range
            except AttributeError:
                raise AttributeError('data must be loaded with '
                    'sp8_image.load_stack() prior to '
                    'calling sp8_image.get_dimension_steps() with '
                    'load_stack_indices=True')

            if dim in dim_range:
                if type(dim)==int:
                    dim = _DimID_to_str(dim)
                steps = steps[dim_range[dim]]
        
        return steps,dimdata['Unit']
    
    def get_pixelsize(self):
        """
        shorthand for `get_dimension_stepsize()` to get the pixel/voxel size
        converted to micrometer, along whatever spatial dimensions are present 
        in the data. Is given as (z,y,x) where dimensions not present in the 
        data are skipped.
        
        Returns
        -------
        pixelsize : tuple of float
            physical size in µm of the pixels/voxels along (z,y,x)
        """
        
        pixelsize = []
        for d in ['z-axis','y-axis','x-axis']:
            try:
                pixelsize.append(self.get_dimension_stepsize(d)[0]*1e6)
            except ValueError:
                pass
        return tuple(pixelsize)
    
    def load_frame(self,i=0,channel=None):
        """
        returns specified image frame where a frame is considered a 2D image in
        the plane of the two fastes axes in the recording order (typically xy).
                
        Parameters
        ----------
        i : int, optional
            the index number of the requested image. The default is 0.
        channel : int or list of int, optional
            which channel(s) to return. For multiple channels, a tuple with an 
            numpy.ndarray for each image is returned, for a single channel a 
            single numpy.ndarray is returned. The default is to return all 
            channels.
        
        Returns
        -------
        frame : numpy.ndarray or tuple of numpy.ndarray
            the raw image data values for the requested frame / channel(s)
        """
        
        #check image index
        if i>=len(self):
            raise IndexError('requested image out of range')
        
        #get default channel
        if channel is None:
            if self._is_multichannel:
                channel = range(self.channels)
            else:
                channel = 0
        elif not self._is_multichannel and channel != 0:
            raise IndexError('requested channel not in data')
        
        #get dims
        dims = self.get_dimensions()

        #for just a single image, we just load it
        if len(dims)==2:
            dimsdict = None
        #for 3D data the i-th frame is just the i-th im along the third dim
        elif len(dims)==3:
            dimsdict = {int(dims[2]['DimID']):i}
        #for higher dims, we need to separate i into the components along each
        #dimension beyond the first 2, by floor dividing out lower dims and
        #and modulo-ing the current dim with its length
        else:
            dimsdict = dict()
            div = 1
            for d in dims[2:]:
                mod = int(d['NumberOfElements'])
                dimsdict[int(d['DimID'])] = (i//div) % mod
                div *= mod
        
        #return requested image with get_plane and correct dimension(s)
        if isinstance(channel,int):
            return self.lifimage.get_plane(c=channel,requested_dims=dimsdict)
        else:
            return tuple([self.lifimage.get_plane(c=c,requested_dims=dimsdict)\
                          for c in channel])
    
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
              
            0. `'channel'` (excluded for single channel data)
            
            1. `'x-axis'`
            
            2. `'y-axis'`
            
            3. `'z-axis'`
            
            4. `'time'`
            
            5. `'detection wavelength'`
            
            9. `'excitation wavelength'`
            
            10. `'mosaic'`
        
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
        order = ['mosaic','channel','time','z-axis','y-axis','x-axis']
        
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
        msteps = np.array(range(self.dims.m))[dim_range['mosaic']]
        
        #determine shape and init array
        newshape = (len(msteps),len(channels),len(times),len(zsteps),ny,nx)
        data = np.empty(newshape,dtype=np.uint8)

        #loop over indices and load
        for i,m in enumerate(msteps):
            for j,c in enumerate(channels):
                for k,t in enumerate(times):
                    for l,z in enumerate(zsteps):
                        data[i,j,k,l,:,:] = self.lifimage.get_frame(z,t,c,m)
        
        #if ranges for x or y are chosen, remove those from the array now,
        #account (top to bottom) for trimming x ánd y, only x, or only y.
        if 'x-axis' in dim_range:
            if 'y-axis' in dim_range:
                slices = tuple([slice(None)]*len(newshape[:-2]) + 
                               [dim_range['y-axis'],dim_range['x-axis']])
            else:
                slices = tuple([slice(None)]*len(newshape[:-1]) + 
                               [dim_range['x-axis']])
            data = data[slices]
        elif 'y-axis' in dim_range:
            slices = tuple([slice(None)]*len(newshape[:-2]) + 
                           [dim_range['y-axis']])
            data = data[slices]
        
        #squeeze out dimensions with only one element
        dim_order = []
        for i,s in enumerate(data.shape):
            if s > 1:
                dim_order.append(order[i])
        data = np.squeeze(data)

        return data, tuple(dim_order)
    
    def export_with_scalebar(self,frame=0,channel=0,filename=None,**kwargs):
        """
        saves an exported image of the confocal slice with a scalebar in one of
        the four corners, where barsize is the scalebar size in data units 
        (e.g. µm) and scale the overall size of the scalebar and text with 
        respect to the width of the image. Additionally, a colormap is applied
        to the data for better visualisation.

        Parameters
        ----------
        frame : int, optional
            index of the frame to export. The default is 0.
        channel : int or list of int, optional
            the channel to pull the image data from. For displaying multiple 
            channels in a single image, a list of channel indices can be given,
            as well as a list of colormaps for each channel through the `cmap` 
            parameter. The default is `0`.
        filename : string or `None`, optional
            Filename + extension to use for the export file. The default is the
            filename sans extension of the original TEM file, with 
            '_exported.png' appended.
        crop : tuple or `None`, optional 
            range describing a area of the original image (before rescaling the
            resolution) to crop out for the export image. Can have two forms:
                
            - `((xmin,ymin),(xmax,ymax))`, with the integer indices of the top
            left and bottom right corners respectively.
                
            - `(xmin,ymin,w,h)` with the integer indices of the top left corner
            and the width and heigth of the cropped image in pixels (prior to 
            optional rescaling using `resolution`).
            
            The default is `None` which takes the entire image.
        cmap : str or callable or list of str or list of callable, optional
            name of a named Matplotlib colormap used to color the data. see the 
            [Matplotlib documentation](
                https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            for more information. The default is `'inferno'`.
            
            In addition to the colormaps listed there, the following maps for 
            linearly incrementing pure RGB channels are available, useful for 
            e.g. displaying multichannel data with complementary colors (no 
            overlap between between colormaps possible):
            ```
            ['pure_reds', 'pure_greens', 'pure_blues', 'pure_yellows', 
             'pure_cyans', 'pure_purples','pure_greys']
            ```
            where for example `'pure_reds'` scales between RGB values `(0,0,0)`
            and  `(255,0,0)`, and `'pure_cyans'` between `(0,0,0)` and 
            `(0,255,255)`.
            
            Alternatively, a fully custom colormap may be used by entering a 
            [ListedColormap](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html#matplotlib.colors.ListedColormap)
            or [LinearSegmentedColormap](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html#matplotlib.colors.LinearSegmentedColormap)
            object from the Matplotlib.colors module. For more information on 
            creating colormaps, see the Matplotlib documentation linked above.
            
            For multichannel data, a list of colormaps *must* be provided, with
            a separate colormap for each channel.
        resolution : int, optional
            the resolution along the x-axis (i.e. image width in pixels) to use
            for the exported image. The default is `None`, which uses the size 
            of the original image (after optional cropping using `crop`).
        cmap_range : tuple of form (min,max), optional
            sets the scaling of the colormap. The minimum and maximum 
            values to map the colormap to, values outside of this range will
            be colored according to the min and max value of the colormap. The 
            default is to take the lowest and highest value in the image.
        draw_bar : boolean, optional
            whether to draw a scalebar on the image, such that this function 
            may be used just to apply a colormap. The default is `True`.
        barsize : float or `None`, optional
            size (in data units matching the original scale bar, e.g. nm) of 
            the scale bar to use. The default `None`, wich takes the desired 
            length for the current scale (ca. 15% of the width of the image for
            `scale=1`) and round this to the nearest option from a list of 
            "nice" values.
        scale : float, optional
            factor to change the size of the scalebar+text with respect to the
            width of the image. Scale is chosen such, that at `scale=1` the
            font size of the scale bar text is approximately 10 pt when 
            the image is printed at half the width of the text in a typical A4
            paper document (e.g. two images side-by-side). Note that this is 
            with respect to the **output** image, so after optional cropping 
            and/or up/down sampling has been applied. The default is `1`.
        loc : int, one of [`0`,`1`,`2`,`3`], optional
            Location of the scalebar on the image, where `0`, `1`, `2` and `3` 
            refer to the top left, top right, bottom left and bottom right 
            respectively. The default is `2`, which is the bottom left corner.
        convert : str, one of [`pm`,`nm`,`um`,`µm`,`mm`,`m`], optional
            Unit that will be used for the scale bar, the value will be 
            automatically converted if this unit differs from the pixel size
            unit. The default is `None`, which uses micrometers.
        font : str, optional
            filename of an installed TrueType font ('.ttf' file) to use for the
            text on the scalebar. The default is `'arialbd.ttf'`.
        fontsize : int, optional
            base font size to use for the scale bar text. The default is 16. 
            Note that this size will be re-scaled according to `resolution` and
            `scale`.
        fontbaseline : int, optional
            vertical offset for the baseline of the scale bar text in printer
             points. The default is 0.
        fontpad : int, optional
            minimum size in printer points of the space/padding between the 
            text and the bar and surrounding box. The default is 2.
        barcolor : tuple of ints, optional
            RGB color to use for the scalebar and text, given
            as a tuple of form (R,G,B) where R, G B and A are values between 0 
            and 255 for red, green and blue respectively. The default is 
            `(255,255,255,255)`, which is a white scalebar and text.
        barthickness : int, optional
            thickness in printer points of the scale bar itself. The default is
            16.
        barpad : int, optional
            size in printer points of the padding between the scale bar and the
            surrounding box. The default is 10.
        box : bool, optional
            Whether to put a colored box behind the scalebar and text to 
            enhance contrast on busy images. The default is `False`.
        boxcolor : tuple of ints, optional
            RGB color to use for the box behind/around the scalebar and text,
            given as a tuple of form (R,G,B) where R, G B and A are values 
            between 0 and 255 for red, green and blue respectively. The default
            is (0,0,0) which gives a black box.
        boxopacity : int, optional
            value between 0 and 255 for the opacity/alpha of the box, useful
            for creating a semitransparent box. The default is 255.
        boxpad : int, optional
            size of the space/padding around the box (with respect to the sides
            of the image) in printer points. The default is 10.
        """      
        #check if pixelsize already calculated, otherwise call get_pixelsize
        pixelsize, unit = self.get_dimension_stepsize('x-axis')
        
        #set default export filename
        if type(filename) != str:
            filename = self.get_name()+'_scalebar.png'
        
        #check we're not overwriting the original file
        if filename==self.filename:
            raise ValueError('overwriting original file not recommended, '+
                             'use a different filename for exporting.')
        
        #check if multichannel or not
        if type(channel)==int:
            multichannel = False
        else:
            multichannel = True
            if not self._is_multichannel:
                raise ValueError('cannot set multiple channels for single '
                                 'channel data')
        
        #get dimensionality of the image to calculate which frame to get
        dims = self.lifimage.dims
        
        #get image (single channel) or list of images (multichannel)
        if multichannel:
            exportim = [
                np.array(self.lifimage.get_frame(
                    z=frame%dims.z,
                    t=frame//dims.z,
                    c=ch
                    ))
                for ch in channel
            ]
        else:
            exportim = np.array(self.lifimage.get_frame(
                z=frame%dims.z,
                t=frame//dims.z,
                c=channel
            ))
        
        #call main export_with_scalebar function with correct pixelsize etc
        from .util import _export_with_scalebar
        _export_with_scalebar(exportim, pixelsize, unit, filename, 
                              multichannel, **kwargs)

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
        the image data as loaded on the most recent call of 
        sp8_series.load_data()
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
        """
        #look for images
        self.filenames = glob.glob(fmt)
        if len(self.filenames) < 1:
            raise ValueError('No images found in current directory')
        
        #check number of channels
        self.channels = len(self.get_metadata_channels())
        if self.channels == 1:
            self._is_multichannel = False
        else:
            self._is_multichannel = True
        
    def __len__(self):
        """define length as number of images (where multiple channels do not) 
        contribute to the count"""
        #only calculate length once
        if hasattr(self, '_len'):
            return self._len
        else:
            self._len = len(self.filenames) // self.channels
            return self._len
    
    def __getitem__(self,key):
        """get i-th recorded 2D image (where multiple channels are considered 
        part of the same image), return as numpy array or tuple of numpy arrays
        for multichannel data"""
        #for slice, turn into indices
        if isinstance(key,slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        #for int, check multichannel and return item(s)
        if self._is_multichannel:
            return tuple([im for im in self.load_data(
                self.filenames[key*self.channels:(key+1)*self.channels:])])
        else:
            return self.load_data(self.filenames[key:key+1])[0]
    
    def __iter__(self):
        """initialize iterator"""
        self._iter_n = 0
        return self

    def __next__(self):
        "make iterable where it returns one image at a time"
        self._iter_n += 1
        if self._iter_n >= len(self):
            raise StopIteration
        else:
            return self[self._iter_n]
    
    def __repr__(self):
        """represents class instance in interpreter"""
        return f"scm_confocal.sp8_series('{self.get_series_name()}')"
    
    def __str__(self):
        """for convenience print basic series info"""
        return "<scm_confocal.sp8_series()>\n" +\
            f'name:\t{self.get_series_name()}\n' +\
            f'files:\t{len(self.filenames)}'
        
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
        else:
            for file in filenames:
                if not os.path.exists(file):
                    raise FileNotFoundError(
                        f'could not find the file "{file}"')

        #this ugly try-except block tries different importers for availability
        try:
            #check pillow import
            from PIL.Image import open as imopen
            data = np.array(
                [np.array(imopen(name)) for name in filenames[first:last]]
            )
            data[0]*1
        except:
            print('[WARNING] scm_confocal.load_data: could not import with '+
                  'PIL, retrying with scikit-image. Make sure libtiff version'+
                  ' >= 4.0.10 is installed')
            try:
                from skimage.io import imread
                data = np.array(
                    [imread(name) for name in filenames[first:last]]
                )
            except:
                raise ImportError('could not load data, check if pillow/PIL '+
                                  'or scikit-image are installed')

        #check if images are 2D (i.e. greyscale)
        if data.ndim > 3:
            print("[WARNING] sp8_series.load_data(): images do not have the "+
                  "correct dimensionality, did you load colour images "+
                  "perhaps? Continueing with average values of higher "+
                  "dimensions")
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
          
            0. `'channel'` (excluded for single channel data)
            
            1. `'x-axis'`
            
            2. `'y-axis'`
            
            3. `'z-axis'`
            
            4. `'time'`
            
            5. `'detection wavelength'`
            
            9. `'excitation wavelength'`
             
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
        newshape = [int(dim['NumberOfElements']) \
                    for dim in reversed(dimensions)]
        
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
            
            #reshape the filenames and apply slicing, then ravel back to flat
            #list
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
                slices = tuple([slice(None)]*len(newshape[:-2]) + 
                               [dim_range['y-axis'],dim_range['x-axis']])
            else:
                slices = tuple([slice(None)]*len(newshape[:-1]) + 
                               [dim_range['x-axis']])
            data = data[slices]
        elif 'y-axis' in dim_range:
            slices = tuple([slice(None)]*len(newshape[:-2]) + 
                           [dim_range['y-axis']])
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
        
        metadata_path = sorted(
            glob.glob(os.path.join(os.path.curdir, 'MetaData', '*.xml'))
        )[0]
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
        
        dimensions = [dict(dim.attrib) \
                      for dim in metadata.find('.//Dimensions')]
        
        self.metadata_dimensions = dimensions
        
        #now that we have dimensions, add shape to attributes
        self.shape = (
            len(self.filenames),
            int(dimensions[1]['NumberOfElements']),
            int(dimensions[0]['NumberOfElements'])
        )
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
    
    def export_with_scalebar(self,frame=0,channel=0,filename=None,**kwargs):
        """
        saves an exported image of the confocal slice with a scalebar in one of
        the four corners, where barsize is the scalebar size in data units 
        (e.g. µm) and scale the overall size of the scalebar and text with 
        respect to the width of the image. Additionally, a colormap is applied
        to the data for better visualisation.

        Parameters
        ----------
        frame : int, optional
            index of the frame to export. The default is 0.
        channel : int or list of int, optional
            the channel to pull the image data from. For displaying multiple 
            channels in a single image, a list of channel indices can be given,
            as well as a list of colormaps for each channel through the `cmap` 
            parameter. The default is `0`.
        filename : string or `None`, optional
            Filename + extension to use for the export file. The default is the
            filename sans extension of the original TEM file, with 
            '_exported.png' appended.
        crop : tuple or `None`, optional 
            range describing a area of the original image (before rescaling the
            resolution) to crop out for the export image. Can have two forms:
                
            - `((xmin,ymin),(xmax,ymax))`, with the integer indices of the top
            left and bottom right corners respectively.
                
            - `(xmin,ymin,w,h)` with the integer indices of the top left corner
            and the width and heigth of the cropped image in pixels (prior to 
            optional rescaling using `resolution`).
            
            The default is `None` which takes the entire image.
        cmap : str or callable or list of str or list of callable, optional
            name of a named Matplotlib colormap used to color the data. see the 
            [Matplotlib documentation](
                https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            for more information. The default is `'inferno'`.
            
            In addition to the colormaps listed there, the following maps for 
            linearly incrementing pure RGB channels are available, useful for 
            e.g. displaying multichannel data with complementary colors (no 
            overlap between between colormaps possible):
            ```
            ['pure_reds', 'pure_greens', 'pure_blues', 'pure_yellows', 
             'pure_cyans', 'pure_purples','pure_greys']
            ```
            where for example `'pure_reds'` scales between RGB values `(0,0,0)`
            and  `(255,0,0)`, and `'pure_cyans'` between `(0,0,0)` and 
            `(0,255,255)`.
            
            Alternatively, a fully custom colormap may be used by entering a 
            [ListedColormap](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html#matplotlib.colors.ListedColormap)
            or [LinearSegmentedColormap](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html#matplotlib.colors.LinearSegmentedColormap)
            object from the Matplotlib.colors module. For more information on 
            creating colormaps, see the Matplotlib documentation linked above.
            
            For multichannel data, a list of colormaps *must* be provided, with
            a separate colormap for each channel.
        resolution : int, optional
            the resolution along the x-axis (i.e. image width in pixels) to use
            for the exported image. The default is `None`, which uses the size 
            of the original image (after optional cropping using `crop`).
        cmap_range : tuple of form (min,max), optional
            sets the scaling of the colormap. The minimum and maximum 
            values to map the colormap to, values outside of this range will
            be colored according to the min and max value of the colormap. The 
            default is to take the lowest and highest value in the image.
        draw_bar : boolean, optional
            whether to draw a scalebar on the image, such that this function 
            may be used just to apply a colormap. The default is `True`.
        barsize : float or `None`, optional
            size (in data units matching the original scale bar, e.g. nm) of 
            the scale bar to use. The default `None`, wich takes the desired 
            length for the current scale (ca. 15% of the width of the image for
            `scale=1`) and round this to the nearest option from a list of 
            "nice" values.
        scale : float, optional
            factor to change the size of the scalebar+text with respect to the
            width of the image. Scale is chosen such, that at `scale=1` the
            font size of the scale bar text is approximately 10 pt when 
            the image is printed at half the width of the text in a typical A4
            paper document (e.g. two images side-by-side). Note that this is 
            with respect to the **output** image, so after optional cropping 
            and/or up/down sampling has been applied. The default is `1`.
        loc : int, one of [`0`,`1`,`2`,`3`], optional
            Location of the scalebar on the image, where `0`, `1`, `2` and `3` 
            refer to the top left, top right, bottom left and bottom right 
            respectively. The default is `2`, which is the bottom left corner.
        convert : str, one of [`pm`,`nm`,`um`,`µm`,`mm`,`m`], optional
            Unit that will be used for the scale bar, the value will be 
            automatically converted if this unit differs from the pixel size
            unit. The default is `None`, which uses micrometers.
        font : str, optional
            filename of an installed TrueType font ('.ttf' file) to use for the
            text on the scalebar. The default is `'arialbd.ttf'`.
        fontsize : int, optional
            base font size to use for the scale bar text. The default is 16. 
            Note that this size will be re-scaled according to `resolution` and
            `scale`.
        fontbaseline : int, optional
            vertical offset for the baseline of the scale bar text in printer
             points. The default is 0.
        fontpad : int, optional
            minimum size in printer points of the space/padding between the 
            text and the bar and surrounding box. The default is 2.
        barcolor : tuple of ints, optional
            RGB color to use for the scalebar and text, given
            as a tuple of form (R,G,B) where R, G B and A are values between 0 
            and 255 for red, green and blue respectively. The default is 
            `(255,255,255,255)`, which is a white scalebar and text.
        barthickness : int, optional
            thickness in printer points of the scale bar itself. The default is
            16.
        barpad : int, optional
            size in printer points of the padding between the scale bar and the
            surrounding box. The default is 10.
        box : bool, optional
            Whether to put a colored box behind the scalebar and text to 
            enhance contrast on busy images. The default is `False`.
        boxcolor : tuple of ints, optional
            RGB color to use for the box behind/around the scalebar and text,
            given as a tuple of form (R,G,B) where R, G B and A are values 
            between 0 and 255 for red, green and blue respectively. The default
            is (0,0,0) which gives a black box.
        boxopacity : int, optional
            value between 0 and 255 for the opacity/alpha of the box, useful
            for creating a semitransparent box. The default is 255.
        boxpad : int, optional
            size of the space/padding around the box (with respect to the sides
            of the image) in printer points. The default is 10.
        """    
        #check if pixelsize already calculated, otherwise call get_pixelsize
        pixelsize, unit = self.get_dimension_stepsize('x-axis')
        
        #set default export filename
        if type(filename) != str:
            filename = self.get_series_name()+'_scalebar.png'
        
        #check we're not overwriting the original file
        if filename in self.filenames:
            raise ValueError('overwriting original file not recommended, '+
                             'use a different filename for exporting.')
        
        #check if multichannel or not
        if type(channel)==int:
            multichannel = False
        else:
            multichannel = True
        
        #get dimensionality of the image and use it to calculate which frame 
        #to get
        #get image (single channel) or list of images (multichannel)
        if multichannel:
            to_load = [self.filenames[
                frame*len(self.get_metadata_channels())+ch] \
                    for ch in channel]
            exportim = self.load_data(filenames=to_load)
        else:
            to_load = self.filenames[
                frame*len(self.get_metadata_channels())+channel
            ]
            exportim = self.load_data(filenames=[to_load])[0]
        
        #call main export_with_scalebar function with correct pixelsize etc
        from .util import _export_with_scalebar
        _export_with_scalebar(exportim, pixelsize, unit, filename, 
                              multichannel, **kwargs)

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
