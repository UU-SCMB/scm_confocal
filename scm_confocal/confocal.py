# -*- coding: utf-8 -*-
"""
created by:     Maarten Bransen
email:          m.bransen@uu.nl
last updated:   06-11-2019
"""

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
        
        #replace dimID with more sensible label
        def DimIDreplace(idlist):
            pattern = zip(list('123456'),['x-axis','y-axis','z-axis','time',
                          'detection wavelength','emission wavelength'])
            for l,r in pattern:
                idlist = idlist.replace(l,r)
            return idlist
        
        order = [DimIDreplace(dim['DimID']) for dim in reversed(dimensions)]
        
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

            #change dim order if multiple channels, move to 0th axis
            for i,dim in enumerate(order):
                if dim == 'channel':
                    filenames = np.moveaxis(filenames,i,0)
                    order = [order[i]] + order[:i] + order[i+1:]

            newshape = list(np.shape(filenames)) + newshape[-2:]
            filenames = list(filenames.ravel())

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


import pims

class visitech_series:
    """
    functions for image series taken with the multi-D acquisition menue in 
    MicroManager with the Visitech saved to multipage .ome.tiff files. For the
    custom fast stack sequence use visitech_faststack.
    """
    def __init__(self,filename,magnification=63,binning=1):
        """
        initialize class (lazy-loads data)

        Parameters
        ----------
        filenames : string
            name of first ome.tiff file (extension optional)
        magnification : float, optional
            magnification of objective lens used. The default is 63.
        binning : int
            binning factor performed at the detector level, e.g. in
            MicroManager software, in XY

        """

        self.filename = filename

        #lazy-load data using PIMS
        print('initializing visitech_series')
        self.datafile = pims.TiffStack(filename)

        #find logical sizes of data
        self.nf = len(self.datafile)

        #find physical sizes of data
        self.magnification = magnification
        self.binning = binning
        self._pixelsizeXY = 6.5/magnification*binning
        #Hamamatsu C11440-22CU has pixels of 6.5x6.5 um

    def load_data(self,indices=slice(None,None,None),dtype=np.uint16):
        """
        load images from datafile into 3D numpy array

        Parameters
        ----------
        indices : slice object or list of ints, optional
            which images from tiffstack to load. The default is
            slice(None,None,None).
        dtype : np int datatype
            data type / bit depth to rescale data to.

        Returns
        -------
        numpy.ndarray containing image data in dim order (im,y,x)

        """
        if type(indices) == slice:
            indices = range(self.nf)[indices]

        data = np.array(self.datafile[indices])

        if not data.dtype == dtype:
            print('rescaling data to type',dtype)
            datamin = np.amin(data)
            data = (data-datamin)/(np.amax(data)-datamin)*np.iinfo(dtype).max
            data = data.astype(dtype)

        return data

    def load_stack(self,dim_range={},dtype=np.uint16):
        """
        Load the data and reshape into 4D stack with the following dimension
        order: ('channel','time','z-axis','y-axis','x-axis') where dimensions
        with len 1 are omitted.
        
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
            the data along any of the dimensions, such as only loading two
            time steps or a particular z-range. An example use for only taking
            time steps up to 5 and z-slice 20 to 30 would
            be:
                dim_range={'time':slice(None,5), 'z-axis':slice(20,30)}.
            The default is {} which corresponds to the full file.
        dtype : (numpy) datatype, optional
            type to scale data to. The default is np.uint16.
        remove_backsteps : bool
            whether to discard the frames which were recorded on the backsteps
            downwards

        Returns
        -------
        data : numpy.ndarray
            ndarray with the pixel values

        """
        #load the stack shape from metadata or reuse previous result
        try:
            self.shape
        except AttributeError:
            self.get_metadata_dimensions()

        #find shape and reshape indices
        shape = self.shape
        if 'x-axis' in self.dimensions:
            shape = shape[:-1]
        if 'y-axis' in self.dimensions:
            shape = shape[:-1]

        indices = np.reshape(range(self.nf),shape)

        #check dim_range items for faulty values
        for key in dim_range.keys():
            if type(key) != str or key not in self.dimensions:
                print("[WARNING] confocal.visitech_faststack.load_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)

        #warn for inefficient x and y trimming
        if 'x-axis' in dim_range or 'y-axis' in dim_range:
            print("[WARNING] confocal.visitech_faststack.load_stack: Loading"+
                  " only part of the data along dimensions 'x-axis' and/or "+
                  "'y-axis' not implemented. Data will be loaded fully "+
                  "into memory before discarding values outside of the "+
                  "slice range specified for the x-axis and/or y-axis. "+
                  "Other axes for which a range is specified will still "+
                  "be treated normally, avoiding unneccesary memory use.")

        #remove values outside of dim_range from indices
        if 'time' in dim_range:
            #this enumerate/tuple construction assures we slice the correct dim
            for i,dim in enumerate(self.dimensions):
                if dim=='time':
                    indices = indices[(slice(None),)*i+(dim_range['time'],)]
        if 'z-axis' in dim_range:
            for i,dim in enumerate(self.dimensions):
                if dim=='z-axis':
                    indices = indices[(slice(None),)*i+(dim_range['z-axis'],)]

        #store image indices array for self.get_timestamps(load_stack_indices=True)
        self._stack_indices = indices

        #load and reshape data
        stack = self.load_data(indices=indices.ravel(),dtype=dtype)
        shape = indices.shape+stack.shape[-2:]
        stack = stack.reshape(shape)

        #trim x and y axis
        if 'y-axis' in dim_range:
            for i,dim in enumerate(self.dimensions):
                if dim=='y-axis':
                    stack = stack[(slice(None),)*i+(dim_range['y-axis'],)]
        if 'x-axis' in dim_range:
            for i,dim in enumerate(self.dimensions):
                if dim=='x-axis':
                    stack = stack[(slice(None),)*i+(dim_range['x-axis'],)]

        return stack

    def yield_stack(self,dim_range={},dtype=np.uint16,remove_backsteps=True):
        """
        Lazy-load the data and reshape into 4D stack with the following
        dimension order: ('time','z-axis','y-axis','x-axis'). Returns a
        generator which yields a z-stack for each call, which is loaded upon
        calling it.
        
        For loading only part of the total dataset, the dim_range parameter can
        be used to specify a range along any of the dimensions. This will be
        more memory efficient than loading the entire stack and then discarding
        part of the data. For slicing along the x or y axis this is not
        possible and whole (xy) images must be loaded prior to discarding
        data outside the specified x or y axis range.

        The shape of the stack can be accessed without loading data using the 
        stack_shape attribute after creating the yield_stack object.

        Parameters
        ----------
        dim_range : dict, optional
            dict, with keys corresponding to channel/dimension labels as above
            and slice objects as values. This allows you to only load part of
            the data along any of the dimensions, such as only loading two
            time steps or a particular z-range. An example use for only taking
            time steps up to 5 and z-slice 20 to 30 would
            be:
                dim_range={'time':slice(None,5), 'z-axis':slice(20,30)}.
            The default is {} which corresponds to the full file.
        dtype : (numpy) datatype, optional
            type to scale data to. The default is np.uint16.
        remove_backsteps : bool
            whether to discard the frames which were recorded on the backsteps
            downwards

        Returns
        -------
        zstack : iterable/generator yielding numpy.ndarray
            list of time steps, with for each time step a z-stack as np.ndarray
            with the pixel values

        """
        indices = np.reshape(range(self.nf),(self.nt,self.nz+self.backsteps))

        #remove backsteps from indices
        if remove_backsteps:
            indices = indices[:,:self.nz]

        #check dim_range items for faulty values
        for key in dim_range.keys():
            if type(key) != str or key not in ['time','z-axis','y-axis','x-axis']:
                print("[WARNING] confocal.visitech_faststack.load_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)

        #warn for inefficient x and y trimming
        if 'x-axis' in dim_range or 'y-axis' in dim_range:
            print("[WARNING] confocal.visitech_faststack.load_stack: Loading"+
                  " only part of the data along dimensions 'x-axis' and/or "+
                  "'y-axis' not implemented. Data will be loaded fully "+
                  "into memory before discarding values outside of the "+
                  "slice range specified for the x-axis and/or y-axis. "+
                  "Other axes for which a range is specified will still "+
                  "be treated normally, avoiding unneccesary memory use.")

        #remove values outside of dim_range from indices
        if 'time' in dim_range:
            indices = indices[dim_range['time']]
        if 'z-axis' in dim_range:
            #assure backsteps cannot be removed this way
            if remove_backsteps:
                indices = indices[:,dim_range['z-axis']]
            else:
                backsteps = indices[:,self.nz:]
                indices = indices[:,dim_range['z-axis']]
                indices = np.concatenate((indices,backsteps),axis=1)

        #store image indices array for self.get_timestamps(load_stack_indices=True)
        self._stack_indices = indices

        #store stack size as attribute
        self.stack_shape = indices.shape + self.datafile[0].shape
        if 'y-axis' in dim_range:
            self.stack_shape = self.stack_shape[:2] + (len(range(self.stack_shape[2])[dim_range['y-axis']]),self.stack_shape[3])
        if 'x-axis' in dim_range:
            self.stack_shape = self.stack_shape[:3] + (len(range(self.stack_shape[3])[dim_range['x-axis']]),)

        #generator loop over each time step in a inner function such that the
        #initialization is excecuted up to this point upon creation rather than
        #upon iteration over the loop
        def stack_iter():
            for zstack_indices in indices:
                zstack = self.load_data(indices=zstack_indices.ravel(),dtype=dtype)
                zstack = zstack.reshape(self.stack_shape[1:])
    
                #trim x and y axis
                if 'y-axis' in dim_range:
                    zstack = zstack[:,dim_range['y-axis']]
                if 'x-axis' in dim_range:
                    zstack = zstack[:,:,dim_range['x-axis']]

                yield zstack

        return stack_iter()

    def _get_metadata_string(filename,read_from_end=True):
        """reads out the raw metadata from a file"""

        import io

        if read_from_end:
            #open file
            with io.open(filename, 'r', errors='ignore',encoding='utf8') as file:

                #set number of characters to move at a time
                blocksize=2**12
                overlap = 6

                #set starting position
                block = ''
                file.seek(0,os.SEEK_END)
                here = file.tell()-overlap
                end = here + overlap
                file.seek(here, os.SEEK_SET)

                #move back until OME start tag is found, store end tag position
                while 0 < here and '<?xml' not in block:
                    delta = min(blocksize, here)
                    here -= delta
                    file.seek(here, os.SEEK_SET)
                    block = file.read(delta+overlap)
                    if '</OME' in block:
                        end = here+delta+overlap

                #read until end
                file.seek(here, os.SEEK_SET)
                metadata = file.read(end-here)

        #process from start of the file
        else:
            metadata = ''
            read=False
            with io.open(filename, 'r', errors='ignore',encoding='utf8') as file:
                #read file line by line to avoid loading too much into memory
                for line in file:
                    #start reading on start of OME tiff header, break at end tag
                    if '<OME' in line:
                        read = True
                    if read:
                        metadata += line
                        if '</OME' in line:
                            break

        #cut off extra characters from end
        return metadata[metadata.find('<?xml'):metadata.find('</OME>')+6]

    def get_metadata(self,read_from_end=True):
        """
        loads OME metadata from visitech .ome.tif file and returns xml tree object

        Parameters
        ----------
        read_from_end : bool, optional
            Whether to look for the metadata from the end of the file.
            The default is True.

        Returns
        -------
        xml.etree.ElementTree
            formatted XML metadata. Can be indexed with
            xml_root.find('<element name>')

        """
        import xml.etree.ElementTree as et

        metadata = visitech_faststack._get_metadata_string(self.filename)

        #remove specifications
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/OME/2013-06"','')
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/SA/2013-06"','')
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/OME/2015-01"','')
        metadata = metadata.replace('xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2015-01 http://www.openmicroscopy.org/Schemas/OME/2015-01/ome.xsd"','')
        print(metadata)
        self.metadata = et.fromstring(metadata)
        return self.metadata

    def get_metadata_dimensions(self):
        """
        finds the stack's dimensionality and logical shape based on the
        embedded metadata

        Returns
        -------
        shape : tuple of ints
            logical sizes of the stack
        dimorder : tuple of strings
            order of the dimensions corresponding to the shape

        """
        try:
            self.metadata
        except AttributeError:
            self.get_metadata()

        #get logical sizes from metadata
        dimorder_dict = dict(self.metadata.find('Image').find('Pixels').attrib)

        #check each dimension and append if present in dataset
        if dimorder_dict['SizeC'] == '1':
            dimorder = []
            shape = []
        else:
            dimorder = ['channel']
            shape = [int(dimorder_dict['SizeC'])]

        if dimorder_dict['SizeT'] != '1':
            dimorder.append('time')
            shape.append(int(dimorder_dict['SizeT']))

        if dimorder_dict['SizeZ'] != '1':
            dimorder.append('z-axis')
            shape.append(int(dimorder_dict['SizeZ']))

        if dimorder_dict['SizeY'] != '1':
            dimorder.append('y-axis')
            shape.append(int(dimorder_dict['SizeY']))

        if dimorder_dict['SizeX'] != '1':
            dimorder.append('x-axis')
            shape.append(int(dimorder_dict['SizeX']))

        shape = tuple(shape)
        dimorder = tuple(dimorder)

        self.shape = shape
        self.dimensions = dimorder

        return shape,dimorder

    def get_image_metadata(self,indices=slice(None)):
        """
        loads the part of the metadata containing information about the time,
        position etc. for each frame of the series and returns a dataframe
        indexes by image frame

        Parameters
        ----------
        indices : slice object, optional
            which image frames to load the metadata for. The default is all
            frames.

        Returns
        -------
        imagedata : pandas.DataFrame
            the metadata for the images, indexed by frame number.

        """
        import pandas as pd

        #load metadata (or use previous result if already loaded)
        try:
            self.metadata
        except AttributeError:
            self.get_metadata()

        #select part of the metadata with physical sizes for each image
        planedata = self.metadata.find('Image').find('Pixels').findall('Plane')

        imagedata = pd.DataFrame([dict(p.attrib) for p in planedata[indices]])
        imagedata = imagedata.astype({'DeltaT':float,'ExposureTime':float,'PositionZ':float,'TheC':int,'TheT':int,'TheZ':int})

        self.image_metadata = imagedata
        return imagedata

    def get_pixelsize(self):
        """shortcut to get (z,y,x) pixelsize with unit"""
        try:
            self.dimensions
        except AttributeError:
            self.get_metadata_dimensions()

        pixelsize = []
        if 'z-axis' in self.dimensions:
            pixelsize.append(float(dict(self.metadata.find('Image').find('Pixels').attrib)['PhysicalSizeZ']))
        if 'y-axis' in self.dimensions:
            pixelsize.append(self._pixelsizeXY)
        if 'x-axis' in self.dimensions:
            pixelsize.append(self._pixelsizeXY)

        self.pixelsize = pixelsize
        return (self.pixelsize,'µm')

    def get_dimension_steps(self,dim,use_stack_indices=False):
        try:
            self.dimensions
        except AttributeError:
            self.get_metadata_dimensions()

        if dim not in self.dimensions or dim == 'channel':
            raise NotImplementedError('"'+dim+'" is not a valid dimension for visitech_series.get_dimension_steps()')

        if dim == 'time':
            if use_stack_indices:
                self.get_image_metadata(indices=self._stack_indices)
            else:
                try:
                    self.image_metadata
                except AttributeError:
                    self.get_image_metadata()
            return (np.array(self.image_metadata['DeltaT']),'ms')

        if dim == 'z-axis':
            if use_stack_indices:
                self.get_image_metadata(indices=self._stack_indices)
            else:
                try:
                    self.image_metadata
                except AttributeError:
                    self.get_image_metadata()
            return (np.array(self.image_metadata['PositionZ']),'µm')

        if dim == 'y-axis':
            if 'x-axis' in self.dimensions:
                return (np.arange(0,self.shape[-2]*self._pixelsizeXY,self._pixelsizeXY),'µm')
            else:
                return (np.arange(0,self.shape[-1]*self._pixelsizeXY,self._pixelsizeXY),'µm')

        if dim == 'x-axis':
            return (np.arange(0,self.shape[-1]*self._pixelsizeXY,self._pixelsizeXY),'µm')


class visitech_faststack:
    """
    functions for fast stacks taken with the custom MicroManager Visitech 
    driver, saved to multipage .ome.tiff files containing entire stack
    """

    def __init__(self,filename,zsize,zstep,zbacksteps,zstart=0,magnification=63,binning=1):
        """
        initialize class (lazy-loads data)

        Parameters
        ----------
        filenames : string
            name of first ome.tiff file (extension optional)
        zsize : float
            z size (in um) of stack (first im to last)
        zstep : float
            step size in z
        zbacksteps : int
            number of backwards steps in z direction after each stack
        zstart : float
            actual height of bottom of stack/lowest slice. The default is 0.
        magnification : float, optional
            magnification of objective lens used. The default is 63.
        binning : int
            binning factor performed at the detector level, e.g. in
            MicroManager software, in XY


        """

        self.filename = filename

        #lazy-load data using PIMS
        print('starting PIMS')
        self.datafile = pims.TiffStack(filename)
        print('PIMS initialized')

        #find logical sizes of data
        self.nf = len(self.datafile)
        self.nz = int((zsize - zsize % zstep)/zstep + 1)
        self.nt = self.nf//(self.nz + zbacksteps)
        self.backsteps = zbacksteps

        #find physical sizes of data
        self.binning = binning
        self.zsteps = np.linspace(zstart,zstart+zsize,self.nz,endpoint=True)
        self.pixelsize = (zstep,6.5/magnification*binning,6.5/magnification*binning)
        #Hamamatsu C11440-22CU has pixels of 6.5x6.5 um

    def load_data(self,indices=slice(None,None,None),dtype=np.uint16):
        """
        load images from datafile into 3D numpy array

        Parameters
        ----------
        indices : slice object or list of ints, optional
            which images from tiffstack to load. The default is
            slice(None,None,None).

        Returns
        -------
        numpy.ndarray containing image data in dim order (im,y,x)

        """
        if type(indices) == slice:
            indices = range(self.nf)[indices]

        data = np.array(self.datafile[indices])

        if not data.dtype == dtype:
            print('rescaling data to type',dtype)
            data = data/np.amax(data)*np.iinfo(dtype).max
            data = data.astype(dtype)

        return data

    def load_stack(self,dim_range={},dtype=np.uint16,remove_backsteps=True,offset=0):
        """
        Load the data and reshape into 4D stack with the following dimension
        order: ('time','z-axis','y-axis','x-axis')
        
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
            the data along any of the dimensions, such as only loading two
            time steps or a particular z-range. An example use for only taking
            time steps up to 5 and z-slice 20 to 30 would
            be:
                dim_range={'time':slice(None,5), 'z-axis':slice(20,30)}.
            The default is {} which corresponds to the full file.
        dtype : (numpy) datatype, optional
            type to scale data to. The default is np.uint16.
        remove_backsteps : bool
            whether to discard the frames which were recorded on the backsteps
            downwards
        offset : int
            offset the indices by a constant number of frames in case the first
            im is not the first slice of the first stack

        Returns
        -------
        data : numpy.ndarray
            ndarray with the pixel values

        """
        #account for offset errors in data recording
        if offset == 0:
            indices = np.reshape(range(self.nf),(self.nt,self.nz+self.backsteps))
        else:
            #in case of offset, lose one stack in total (~half at begin and half at end)
            nf = self.nf - (self.nz+self.backsteps)
            nt = self.nt - 1
            indices = np.reshape(range(offset,offset+nf),(nt,self.nz+self.backsteps))

        #remove backsteps from indices
        if remove_backsteps:
            indices = indices[:,:self.nz]

        #check dim_range items for faulty values
        for key in dim_range.keys():
            if type(key) != str or key not in ['time','z-axis','y-axis','x-axis']:
                print("[WARNING] confocal.visitech_faststack.load_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)

        #warn for inefficient x and y trimming
        if 'x-axis' in dim_range or 'y-axis' in dim_range:
            print("[WARNING] confocal.visitech_faststack.load_stack: Loading"+
                  " only part of the data along dimensions 'x-axis' and/or "+
                  "'y-axis' not implemented. Data will be loaded fully "+
                  "into memory before discarding values outside of the "+
                  "slice range specified for the x-axis and/or y-axis. "+
                  "Other axes for which a range is specified will still "+
                  "be treated normally, avoiding unneccesary memory use.")

        #remove values outside of dim_range from indices
        if 'time' in dim_range:
            indices = indices[dim_range['time']]
        if 'z-axis' in dim_range:
            #assure backsteps cannot be removed this way
            if remove_backsteps:
                indices = indices[:,dim_range['z-axis']]
            else:
                backsteps = indices[:,self.nz:]
                indices = indices[:,dim_range['z-axis']]
                indices = np.concatenate((indices,backsteps),axis=1)

        #store image indices array for self.get_timestamps(load_stack_indices=True)
        self._stack_indices = indices

        #load and reshape data
        stack = self.load_data(indices=indices.ravel(),dtype=dtype)
        shape = (indices.shape[0],indices.shape[1],stack.shape[1],stack.shape[2])
        stack = stack.reshape(shape)

        #trim x and y axis
        if 'y-axis' in dim_range:
            stack = stack[:,:,dim_range['y-axis']]
        if 'x-axis' in dim_range:
            stack = stack[:,:,:,dim_range['x-axis']]

        return stack

    def yield_stack(self,dim_range={},dtype=np.uint16,remove_backsteps=True,offset=0):
        """
        Lazy-load the data and reshape into 4D stack with the following
        dimension order: ('time','z-axis','y-axis','x-axis'). Returns a
        generator which yields a z-stack for each call, which is loaded upon
        calling it.
        
        For loading only part of the total dataset, the dim_range parameter can
        be used to specify a range along any of the dimensions. This will be
        more memory efficient than loading the entire stack and then discarding
        part of the data. For slicing along the x or y axis this is not
        possible and whole (xy) images must be loaded prior to discarding
        data outside the specified x or y axis range.

        The shape of the stack can be accessed without loading data using the 
        stack_shape attribute after creating the yield_stack object.

        Parameters
        ----------
        dim_range : dict, optional
            dict, with keys corresponding to channel/dimension labels as above
            and slice objects as values. This allows you to only load part of
            the data along any of the dimensions, such as only loading two
            time steps or a particular z-range. An example use for only taking
            time steps up to 5 and z-slice 20 to 30 would
            be:
                dim_range={'time':slice(None,5), 'z-axis':slice(20,30)}.
            The default is {} which corresponds to the full file.
        dtype : (numpy) datatype, optional
            type to scale data to. The default is np.uint16.
        remove_backsteps : bool
            whether to discard the frames which were recorded on the backsteps
            downwards
        offset : int
            offset the indices by a constant number of frames in case the first
            im is not the first slice of the first stack

        Returns
        -------
        zstack : iterable/generator yielding numpy.ndarray
            list of time steps, with for each time step a z-stack as np.ndarray
            with the pixel values

        """
        #account for offset errors in data recording
        if offset == 0:
            indices = np.reshape(range(self.nf),(self.nt,self.nz+self.backsteps))
        else:
            #in case of offset, lose one stack in total (~half at begin and half at end)
            nf = self.nf - (self.nz+self.backsteps)
            nt = self.nt - 1
            indices = np.reshape(range(offset,offset+nf),(nt,self.nz+self.backsteps))


        #remove backsteps from indices
        if remove_backsteps:
            indices = indices[:,:self.nz]

        #check dim_range items for faulty values
        for key in dim_range.keys():
            if type(key) != str or key not in ['time','z-axis','y-axis','x-axis']:
                print("[WARNING] confocal.visitech_faststack.load_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)

        #warn for inefficient x and y trimming
        if 'x-axis' in dim_range or 'y-axis' in dim_range:
            print("[WARNING] confocal.visitech_faststack.load_stack: Loading"+
                  " only part of the data along dimensions 'x-axis' and/or "+
                  "'y-axis' not implemented. Data will be loaded fully "+
                  "into memory before discarding values outside of the "+
                  "slice range specified for the x-axis and/or y-axis. "+
                  "Other axes for which a range is specified will still "+
                  "be treated normally, avoiding unneccesary memory use.")

        #remove values outside of dim_range from indices
        if 'time' in dim_range:
            indices = indices[dim_range['time']]
        if 'z-axis' in dim_range:
            #assure backsteps cannot be removed this way
            if remove_backsteps:
                indices = indices[:,dim_range['z-axis']]
            else:
                backsteps = indices[:,self.nz:]
                indices = indices[:,dim_range['z-axis']]
                indices = np.concatenate((indices,backsteps),axis=1)

        #store image indices array for self.get_timestamps(load_stack_indices=True)
        self._stack_indices = indices

        #store stack size as attribute
        self.stack_shape = indices.shape + self.datafile[0].shape
        if 'y-axis' in dim_range:
            self.stack_shape = self.stack_shape[:2] + (len(range(self.stack_shape[2])[dim_range['y-axis']]),self.stack_shape[3])
        if 'x-axis' in dim_range:
            self.stack_shape = self.stack_shape[:3] + (len(range(self.stack_shape[3])[dim_range['x-axis']]),)

        #generator loop over each time step in a inner function such that the
        #initialization is excecuted up to this point upon creation rather than
        #upon iteration over the loop
        def stack_iter():
            for zstack_indices in indices:
                zstack = self.load_data(indices=zstack_indices.ravel(),dtype=dtype)
                zstack = zstack.reshape(self.stack_shape[1:])
    
                #trim x and y axis
                if 'y-axis' in dim_range:
                    zstack = zstack[:,dim_range['y-axis']]
                if 'x-axis' in dim_range:
                    zstack = zstack[:,:,dim_range['x-axis']]

                yield zstack

        return stack_iter()

    def save_stack(self,data,filename_prefix='visitech_faststack',sequence_type='multipage'):
        """
        save stacks to tiff files

        Parameters
        ----------
        data : numpy ndarray with 3 or 4 dimensions
            image series pixel values with dimension order (z,y,x) or (t,z,y,x)
        filename_prefix : string, optional
            prefix to use for filename. The time/z-axis index is appended if
            relevant. The default is 'visitech_faststack'.
        sequence_type : string, optional
            The way to store the data. The following options are available:
                * 'image_sequence' : stores as a series of 2D images with time and or frame number appended
                * 'multipage' : store all data in a single multipage tiff file
                * 'multipage_sequence' : stores a multipage tiff file for each time step
            The default is 'multipage'.

        Returns
        -------
        None, but writes file(s) to working directory.

        """
        from PIL import Image
        
        shape = np.shape(data)
        
        #store as series of named 2D images
        if sequence_type == 'image_sequence':
            #for (t,z,y,x)
            if len(shape) == 4:
                for i,t in enumerate(data):
                    for j,im in enumerate(t):
                        filename = filename_prefix + '_t{:03d}_z{:03d}.tif'.format(i,j)
                        Image.fromarray(im).save(filename)
            #for (z,y,x)
            elif len(shape) == 3:
                for i,im in enumerate(data):
                    filename = filename_prefix + '_z{:03d}.tif'.format(i,j)
                    Image.fromarray(im).save(filename)  
            else:
                raise ValueError('data must be 3-dimensional (z,y,x) or 4-dimensional (t,z,y,x)')
            
        #store as single multipage tiff
        elif sequence_type == 'multipage':
            #for (t,z,y,x)
            if len(shape) == 4:
                data = [Image.fromarray(im) for _ in data for im in _]
                data[0].save(filename_prefix+'.tif',append_images=data[1:],save_all=True,)
            #for (z,y,x)
            elif len(shape) == 3:
                data = [Image.fromarray(im) for im in data]
                data[0].save(filename_prefix+'.tif',append_images=data[1:],save_all=True,)
            else:
                raise ValueError('data must be 3-dimensional (z,y,x) or 4-dimensional (t,z,y,x)')
            
        elif sequence_type == 'multipage_sequence':
            if len(shape) == 4:
                for i,t in enumerate(data):
                    t = [Image.fromarray(im) for im in t]
                    t[0].save(filename_prefix+'_t{:03d}.tif'.format(i),append_images=t[1:],save_all=True)
            elif len(shape) == 3:
                print("[WARNING] scm_confocal.faststack.save_stack(): 'multipage_sequence' invalid sequence_type for 3-dimensional data. Saving as option 'multipage' instead")
                data = [Image.fromarray(im) for im in data]
                data[0].save(filename_prefix+'.tif',append_images=data[1:],save_all=True)
            else:
                raise ValueError('data must be 4-dimensional (t,z,y,x)')

        else:
            raise ValueError("invalid option for sequence_type: must be 'image_sequence', 'multipage' or 'multipage_sequence'")
    
    def _get_metadata_string(filename,read_from_end=True):
        """reads out the raw metadata from a file"""

        import io

        if read_from_end:
            #open file
            with io.open(filename, 'r', errors='ignore',encoding='utf8') as file:

                #set number of characters to move at a time
                blocksize=2**12
                overlap = 6

                #set starting position
                block = ''
                file.seek(0,os.SEEK_END)
                here = file.tell()-overlap
                end = here + overlap
                file.seek(here, os.SEEK_SET)

                #move back until OME start tag is found, store end tag position
                while 0 < here and '<?xml' not in block:
                    delta = min(blocksize, here)
                    here -= delta
                    file.seek(here, os.SEEK_SET)
                    block = file.read(delta+overlap)
                    if '</OME' in block:
                        end = here+delta+overlap

                #read until end
                file.seek(here, os.SEEK_SET)
                metadata = file.read(end-here)

        #process from start of the file
        else:
            metadata = ''
            read=False
            with io.open(filename, 'r', errors='ignore',encoding='utf8') as file:
                #read file line by line to avoid loading too much into memory
                for line in file:
                    #start reading on start of OME tiff header, break at end tag
                    if '<OME' in line:
                        read = True
                    if read:
                        metadata += line
                        if '</OME' in line:
                            break

        #cut off extra characters from end
        return metadata[metadata.find('<?xml'):metadata.find('</OME>')+6]

    def get_metadata(self,read_from_end=True):
        """
        loads OME metadata from visitech .ome.tif file and returns xml tree object

        Parameters
        ----------
        read_from_end : bool, optional
            Whether to look for the metadata from the end of the file.
            The default is True.

        Returns
        -------
        xml.etree.ElementTree
            formatted XML metadata. Can be indexed with
            xml_root.find('<element name>')

        """
        import xml.etree.ElementTree as et

        metadata = visitech_faststack._get_metadata_string(self.filename)

        #remove specifications
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/OME/2013-06"','')
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/SA/2013-06"','')
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/OME/2015-01"','')
        metadata = metadata.replace('xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2015-01 http://www.openmicroscopy.org/Schemas/OME/2015-01/ome.xsd"','')

        self.metadata = et.fromstring(metadata)
        return self.metadata

    def get_timestamps(self,load_stack_indices=False):
        """
        loads OME metadata from visitech .ome.tif file and returns timestamps
        
        Parameters
        ----------
        load_stack_indices : boolean
            if True, only returns timestamps from frames which were loaded
            at call to visitech_faststack.load_stack(), and using the same
            dimension order / stack shape
    
        Returns
        -------
        times : numpy (nd)array of floats
            list/stack of timestamps for each of the the frames in the data

        """
        import re

        metadata = visitech_faststack._get_metadata_string(self.filename)

        times = re.findall(r'DeltaT="([0-9]*\.[0-9]*)"',metadata)
        times = np.array([float(t) for t in times])

        if load_stack_indices:
            try:
                indices = self._stack_indices
            except AttributeError:
                raise AttributeError('data must be loaded with '+
                                  'visitech_faststack.load_stack() prior to '+
                                  'calling visitech_faststack.get_timestamps()'
                                  +' with load_stack_indices=True')

            times = times[indices.ravel()].reshape(np.shape(indices))

        self.times = times
        return times

    def get_pixelsize(self):
        """shortcut to get (z,y,x) pixelsize with unit"""
        return (self.pixelsize,'µm')

class util:
    """
    Set of utility functions for dealing with stacks and confocal data
    """
    
    def bin_stack(images,n=1,blocksize=None,quiet=False,dtype=np.uint8):
        """
        bins numpy ndarrays in arbitrary dimensions by a factor n. Prior to
        binning, elements from the end are deleted until the length is a
        multiple of the bin factor. Executes averaging of bins in floating
        point precision, which is memory intensive for large stacks. Using
        smaller blocks reduces memory usage, but is less efficient.

        Parameters
        ----------
        images : numpy.ndarray
            ndarray containing the data
        n : int or tuple of int, optional
            factor to bin with for all dimensions (int) or each dimension
            individually (tuple with one int per dimension). The default is 1.
        blocksize : int, optional
            number of (binned) slices to process at a time to conserve memory.
            The default is entire stack.
        quiet : bool, optional
            suppresses printed output when True. The default is False.
        dtype : (numpy) datatype, optional
            datatype to use for output. Averaging of the binned pixels always
            occurs in floating point precision. The default is np.uint8.

        Returns
        -------
        images : numpy.ndarray
            binned stack

        """
        dims = images.ndim
        
        #check input parameters for type
        if type(n) != tuple and type(n) != int:
            print('n must be int or tuple of ints, skipping binning step')
            return images
        
        #when n is int, convert to tuple of ints with length dims
        if type(n) == int:
            n = (n,)*dims
            
        #else if n is a tuple, check if length of n matches dims
        elif len(n) != dims:
            print('number of dimensions does not match, skipping binning step')
            return images
        
        #skip rest of code when every element in n is 1
        if all([nitem==1 for nitem in n]):
            if not quiet:
                print('no binning used')
            return images
    
        #define new shapes
        oldshape = np.shape(images)
        trimmedshape = tuple([int(oldshape[d] - oldshape[d] % n[d]) for d in range(dims)])
        newshape = tuple([int(trimmedshape[d]/n[d]) for d in range(dims)])
        
        #trim ends when new shape is not a whole multiple of binfactor
        slices = [range(trimmedshape[d],oldshape[d]) for d in range(dims)]
        for d in range(dims):
            if trimmedshape[d] != oldshape[d]:
                images = np.delete(images,slices[d],axis=d)
        
        #print old and new shapes when trimming and or binning is used
        if oldshape != trimmedshape and not quiet:
            print('trimming from shape {} to {}, binning to {}...'.format(oldshape,trimmedshape,newshape))
        elif not quiet:
            print('binning from shape {} to {}...'.format(trimmedshape,newshape))
        
        #check block size
        if blocksize == None:
            blocksize = newshape[0]
        
        #determine shape such that reshaped array has dims*2 dimensions, with each dim spread over new axis to bin along
        reshapeshape = [i for subtuple in [(newshape[d],n[d]) for d in range(dims)] for i in subtuple]
        
        #when binning entire array at once
        if blocksize >= newshape[0]:
            #execute reshape
            images = images.reshape(reshapeshape)
            
            #average along binning axes to obtain original dimensionality
            #(floating point precision and memory intensive!!)
            for d in reversed(range(1,2*dims,2)):
                images = images.mean(d)
            
            #set type back from floating point
            images = images.astype(dtype)
            
        #when splitting binning into multiple blocks to conserve memory
        else:
            if not quiet:
                print('splitting binning in {} blocks'.format(int(np.ceil(newshape[0]/blocksize))))
            
            #reshape stack and store to temp variable
            reshapeim = images.reshape(reshapeshape)
            
            #overwrite images with empty array of new shape
            images = np.zeros(newshape,dtype=dtype)
            
            #perform binning in steps of blocksize
            for i in range(0,newshape[0] - (newshape[0] % blocksize),blocksize):
                block = reshapeim[i:i+blocksize]
                for d in reversed(range(1,2*dims,2)):
                    block = np.mean(block,axis=d)
                images[i:i+blocksize] = block.astype(dtype)
            
            #execute last (smaller) block if newshape is not devisible by blocksize
            if newshape[0] % blocksize != 0:
                block = reshapeim[-(newshape[0] % blocksize):]
                for d in reversed(range(1,2*dims,2)):
                    block = np.mean(block,axis=d)
                images[-(newshape[0] % blocksize):] = block.astype(dtype)
        
        if not quiet:
            print('binning finished')
        
        return images
    
    def fit_powerlaw(x,y,weights=None):
        """
        Linear regression in log space of the MSD to get diffusion constant, which
        is a powerlaw in linear space of the form A*x**n

        Parameters
        ----------
        x : list or numpy.array
            x coordinates of data points to fit
        y : list or numpy.array
            y coordinates of data points to fit
        weights : list or numpy.array, optional
            list of weights to use for each (x,y) coordinate. The default is 
            None.

        Returns
        -------
        A : float
            constant A
        n : float
            power n
        sigmaA : float
            standard deviation in A
        sigmaN : float
            standard deviation in n

        """
        import scipy.optimize
        
        def f(x,a,b):
            return a*x + b
        
        if weights is None:
            weights = np.ones(len(y))
        else:
            weights = np.array(weights)
        
        #remove nan values
        x = np.array(x)
        y = np.array(y)
        
        x = x[~np.isnan(y)]
        weights = weights[~np.isnan(y)]
        y = y[~np.isnan(y)]
        
        #fit
        (n,A), covariance = scipy.optimize.curve_fit(f,np.log(x),np.log(y),sigma=weights)
        sigmaN,sigmaA = np.sqrt(np.diag(covariance))
        A = np.exp(A)
        sigmaA = sigmaA*np.exp(A)
        
        return A,n,sigmaA,sigmaN
    
    def mean_square_displacement(features, pos_cols = ['x','y','z'], t_col='t (s)',
                             nparticles=None, pickrandom=False, nbins=20,
                             tmin=None, tmax=None):
        """
        calculate the mean square displacement vs time for linked particles

        Parameters
        ----------
        features : pandas.DataFrame
            output from trackpy.link containing tracking data
        pos_cols : list of str, optional
            names of columns to use for coordinates. The default is
            ['x','y','z'].
        t_col : str, optional
            name of column containing timestamps. The default is 't (s)'.
        nparticles : int, optional
            number of particles to use for calculation (useful for large
            datasets). The default is all particles.
        pickrandom : bool, optional
            whether to pick nparticles randomly or not, if False it takes the 
            n longest tracked particles from data. The default is False.
        nbins : int, optional
            number of bins for output. The default is 20.
        tmin : float, optional
            left edge of first bin. The default is min(t_col).
        tmax : float, optional
           right edge of last bin, The default is max(t_col).

        Returns
        -------
        binedges : numpy.array
            edges of time bins
        bincounts : numpy.array
            number of sampling points for each bin
        binmeans : numpy.array
            mean square displacement values

        """
        #restructure data to correct order
        features = features[['particle']+[t_col]+pos_cols]
        features = features.set_index('particle')
        dims = len(pos_cols)
        
        #initialize empty array to contain [[dt1,dr1],[dt2,dr2],...]
        dt_dr = np.empty((0,2))
        
        #converting to set assures unique values only
        particles = set(features.index)
        
        #optionally take a subset of particles
        if nparticles != None and len(particles)>nparticles:
            
            #optionally take random subset of particles
            if pickrandom:
                import random
                particles = random.sample(set(features.index),nparticles)
            
            #else take the particles which occur in most of the frames
            else:
                vals, counts = np.unique(features.index, return_counts=True)
                sortedindices = np.argsort(-counts)[:nparticles]
                particles = vals[sortedindices]
        
        #iterate over all particles and all time intervals for that particle and 
        # append [[dr,dt]] to dt_dr each time
        for p in particles:
            pdata = features.loc[p]
            for j in range(len(pdata)):
                for i in range(j):
                    dt_dr = np.append(
                            dt_dr,
                            [[
                                    pdata.iat[j,0] - pdata.iat[i,0],
                                    sum([(pdata.iat[j,d] - pdata.iat[i,d])**2 for d in range(1,dims+1)])
                            ]],
                            axis = 0
                            )
        
        #check bins
        if tmin == None:
            tmin = min(dt_dr[:,0])
        if tmax == None:
            tmax = max(dt_dr[:,0])
        
        #create bin edges
        binedges = np.linspace(tmin,tmax,nbins+1,endpoint=True)
        
        #put each timestep into the correct bin and remove data left of first bin
        binpos = binedges.searchsorted(dt_dr[:,0],side='right')
        dt_dr = dt_dr[binpos!=0]
        binpos = binpos[binpos!=0]
        binpos = binpos-1
        
        #count each bin and weigh counts by corresponding r, then normalize by unweighted counts
        bincounts = np.bincount(binpos, minlength=nbins)
        binmeans =  np.bincount(binpos, weights=dt_dr[:,1], minlength=nbins) / bincounts
        
        return binedges,bincounts[:-1],binmeans[:-1]
    
    def mean_square_displacement_per_frame(features, pos_cols = ['x','y'], feat_col = 'particle'):
        """
        Calculate the mean square movement of all tracked features between
        subsequent frames using efficient pandas linear algebra

        Parameters
        ----------
        features : pandas.Dataframe
            dataframe containing the tracking data over timesteps indexed by
            frame number and containing coordinates of features.
        pos_cols : list of str, optional
            names of the columns containing coordinates. The default is
            ['x','y'].
        feat_col : str
            name of column containing feature identifyers. The default is
            'particle'.

        Returns
        -------
        msd : numpy.array
            averages of the squared displacements between each two steps

        """

        nf = int(max(features.index))
        
        features = features[[feat_col]+pos_cols]
        msd = np.empty((nf))
        
        #loop over all sets of subsequent frames 
        for i in range(nf):
            
            #create subsets for current and next frame
            a = features.loc[i].set_index(feat_col)
            b = features.loc[i+1].set_index(feat_col)
            
            #find all features which occur in both frames
            f = set(a.index).intersection(set(b.index))
            
            #join positional columns of particles to a single DataFrame
            a = a.loc[f][pos_cols].join(b.loc[f][pos_cols], rsuffix='_b')
            
            #create a new column with sum of squared displacement along each direction
            a['dr**2'] = np.sum([(a[pos + '_b'] - a[pos])**2 for pos in pos_cols], 0)
            
            #take the mean of all square displacement column and add to list
            msd[i] = a['dr**2'].mean()
        
        return msd

    def subtract_background(images, val=0, percentile=False):
        """
        subtract a constant value from a numpy array without going below 0

        Parameters
        ----------
        images : numpy ndarray
            images to correct.
        percentile : bool, optional
            Whether to give the value as a percentile of the stack rather than
            an absolute value to subtrackt. The default is False.
        val : int or float, optional
            Value or percentile to subtract. The default is 0.

        Returns
        -------
        images : numpy ndarray
            the corrected stack.

        """

        #calculate percentile val
        if percentile:
            val = np.percentile(images,val)

        #correct intensity
        images[images<val] = 0
        images[images>=val] = images[images>=val]-val

        return images

    def plot_stack_histogram(images,bin_edges=range(0,256),newfig=True,legendname=None,title='intensity histogram'):
        """
        manually flattens list of images to list of pixel values and plots
        histogram. Can combine multiple calls with newfig and legendname
        options

        Parameters
        ----------
        images : numpy ndarray
            array containing pixel values
        bin_edges : list or range, optional
            edges of bins to use. The default is range(0,256).
        newfig : bool, optional
            Whether to open a new figure or to add to currently active figure.
            The default is True.
        legendname : string, optional
            label to use for the legend. The default is None.
        title : string, optional
            text to use as plot title. The default is 'intensity histogram'.

        Returns
        -------
        pyplot figure handle

        """
        from matplotlib import pyplot as plt
        
        if newfig:
            fig = plt.figure()
            plt.xlabel('grey value')
            plt.ylabel('counts')
            plt.title(title)
        else:
            fig = plt.gcf()
        
        plt.hist(np.ravel(images),log=(False,True),bins=bin_edges,label=legendname)
        
        if not legendname == None:
            plt.legend()
        plt.show()
        return fig
    
    def multiply_intensity(data,factor,dtype=None):
        """
        For multiplying the values of a numpy array while accounting for
        integer overflow issues in integer datatypes. Corrected values larger
        than the datatype max are set to the max value.

        Parameters
        ----------
        data : numpy.ndarray
            array containing the data values
        factor : float
            factor to multiply data with
        dtype : (numpy) datatype, optional
            Datatype to scale data to. The default is the same type as the
            input data.

        Returns
        -------
        data : numpy.ndarray
            data with new intensity values.

        """

        if factor == 1:
            return data
        
        if dtype == None:
            dtype = data.dtype
        
        #try if integer type, if not it must be float so one can just multiply
        try:
            maxval = np.iinfo(dtype).max
        except ValueError:
            return data*factor
        
        #for integer types, account for integer overflow
        data[data >= maxval/factor] = maxval
        data[data <  maxval/factor] = data[data <  maxval/factor] * factor
        
        return data
    
    def saveprompt(question="Save/overwrite? 1=YES, 0=NO. "):
        """
        aks user to save, returns boolean
        """
        try:
            savefile = int(input(question))
        except ValueError:
            savefile = 0
        if savefile>1 or savefile<0:
            savefile = 0
        if savefile==1:
             print("saving data")
             save = True
        else:
            print("not saving input parameters")
            save = False
        return save
    
    def write_textfile(params,filename="parameters.txt"):
        """
        stores parameter names and values in text file

        Parameters
        ----------
        params : dictionary of name:value
            the data to store
        filename : str, optional
            file name to us for saving. The default is "parameters.txt".

        Returns
        -------
        None.

        """

        with open(filename,'w') as file:
            for key,val in params.items():
                file.write(str(key)+' = '+str(val)+'\n')
        print("input parameters saved in",filename)
