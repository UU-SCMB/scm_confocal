import numpy as np
import pims
from slicerator import Slicerator
import os
from decimal import Decimal
import warnings

class visitech_series:
    """
    Functions for image series taken with the multi-D acquisition menu in 
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

        #find physical sizes of data
        self.magnification = magnification
        self.binning = binning
        self._pixelsizeXY = 6.5/magnification*binning
        #Hamamatsu C11440-22CU has pixels of 6.5x6.5 um
    
    def __repr__(self):
        """"represents class instance in the interpreter"""
        return f"<scm_confocal.visitech_series('{self.filename}')>"

    def __len__(self):
        """length of series is number of frames"""
        if hasattr(self,'nf'):
            return self.nf
        else:
            self._init_pims()
            return self.nf
    
    def __getitem__(self,key):
        """make indexable, where it returns the ith frame, where a frame is 
        defined by the first 2 dimensions in recording order"""
        if not hasattr(self,'nf'):
            self._init_pims()
        if isinstance(key,slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        else:
            return self.datafile[key]
    
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
            return self.datafile[self._iter_n]

    def _init_pims(self):
        """only initialize PIMS object when necessary, since it may take a long
        time for large series / slow harddrives even to just index the data"""
        
        #lazy-load data using PIMS
        print('initializing visitech_series')
        self.datafile = pims.TiffStack(self.filename)

        #find logical sizes of data
        self.nf = len(self.datafile)

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
        #try if PIMS is initialized, if not do so
        try:
            self.nf
        except AttributeError:
            self._init_pims()
        
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
        #try if PIMS is initialized, if not do so
        try:
            self.nf
        except AttributeError:
            self._init_pims()
        
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

        #check dim_range items for faulty values and remove None slices
        for key,val in dim_range.copy().items():
            if type(key) != str or key not in self.dimensions:
                print("[WARNING] confocal.visitech_faststack.load_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)
            elif val==slice(None):
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

        #store image indices array for 
        # self.get_timestamps(load_stack_indices=True)
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
        #try if PIMS is initialized, if not do so
        try:
            self.nf
        except AttributeError:
            self._init_pims()
        
        indices = np.reshape(range(self.nf),(self.nt,self.nz+self.backsteps))

        #remove backsteps from indices
        if remove_backsteps:
            indices = indices[:,:self.nz]

        #check dim_range items for faulty values and remove None slices
        for key,val in dim_range.copy().items():
            if type(key) != str or key not in self.dimensions:
                print("[WARNING] confocal.visitech_faststack.load_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)
            elif val==slice(None):
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

        #store image indices array for
        # self.get_timestamps(load_stack_indices=True)
        self._stack_indices = indices

        #store stack size as attribute
        self.stack_shape = indices.shape + self.datafile[0].shape
        if 'y-axis' in dim_range:
            self.stack_shape = self.stack_shape[:2] + (
                len(range(self.stack_shape[2])[dim_range['y-axis']]),
                self.stack_shape[3]
            )
        if 'x-axis' in dim_range:
            self.stack_shape = self.stack_shape[:3] + \
                (len(range(self.stack_shape[3])[dim_range['x-axis']]),)

        #generator loop over each time step in a inner function such that the
        #initialization is excecuted up to this point upon creation rather than
        #upon iteration over the loop
        def stack_iter():
            for zstack_indices in indices:
                zstack = self.load_data(indices=zstack_indices.ravel(),
                                        dtype=dtype)
                zstack = zstack.reshape(self.stack_shape[1:])
    
                #trim x and y axis
                if 'y-axis' in dim_range:
                    zstack = zstack[:,dim_range['y-axis']]
                if 'x-axis' in dim_range:
                    zstack = zstack[:,:,dim_range['x-axis']]

                yield zstack

        return stack_iter()

    def _get_metadata_string(self):
        """reads out the raw metadata from a file"""
        import struct
        
        #open file  with bytes
        with open(self.filename,'rb') as file:
            
            #find position of Image File Directory (IDF), given as a 32 bit 
            #unsigned integer in byte position 4 in the file. See TIFF 
            #specification: http://www.exif.org/TIFF6.pdf
            file.seek(4)
            IDF_pos = struct.unpack('I',file.read(4))[0]
            
            #first two bytes are number of entries in IDF, then each entry is 
            #12 bytes long. Per micromanager standard, we need the 5th entry 
            #for OME metadata
            #https://micro-manager.org/wiki/Micro-Manager_Image_File_Stacks
            #bytes 4-7 give item length, bytes 8-11 give byte offset of item as
            #per TIFF standard
            file.seek(IDF_pos+2+5*12+4)
            OME_len = struct.unpack('I',file.read(4))[0]
            OME_offset = struct.unpack('I',file.read(4))[0]
            
            #go to offset and read in the desired number of bytes
            file.seek(OME_offset)
            metadata = file.read(OME_len)

        #decode bytes to string
        metadata = metadata.decode('utf-8')
        
        #cut off extra characters from end
        return metadata[metadata.find('<?xml'):metadata.find('</OME>')+6]

    def get_metadata(self):
        """
        loads OME metadata from visitech .ome.tif file and returns xml tree 
        object
            
        Returns
        -------
        xml.etree.ElementTree
            formatted XML metadata. Can be indexed with
            xml_root.find('<element name>')
        """
        import xml.etree.ElementTree as et

        metadata = self._get_metadata_string()

        #remove specifications
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/OME/2013-06"','')
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/SA/2013-06"','')
        metadata = metadata.replace('xmlns="http://www.openmicroscopy.org/Schemas/OME/2015-01"','')
        metadata = metadata.replace('xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2015-01 http://www.openmicroscopy.org/Schemas/OME/2015-01/ome.xsd"','')
        
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
        imagedata = imagedata.astype({'DeltaT':float,'ExposureTime':float,
                                      'PositionZ':float,'TheC':int,'TheT':int,
                                      'TheZ':int})

        self.image_metadata = imagedata
        return imagedata

    def get_pixelsize(self):
        """shortcut to get `(z,y,x)` pixelsize with unit"""
        try:
            self.dimensions
        except AttributeError:
            self.get_metadata_dimensions()

        pixelsize = []
        if 'z-axis' in self.dimensions:
            pixelsize.append(float(dict(
                self.metadata.find('Image').find('Pixels').attrib
            )['PhysicalSizeZ']))
        if 'y-axis' in self.dimensions:
            pixelsize.append(self._pixelsizeXY)
        if 'x-axis' in self.dimensions:
            pixelsize.append(self._pixelsizeXY)

        self.pixelsize = pixelsize
        return (self.pixelsize,'µm')

    def get_dimension_steps(self,dim,use_stack_indices=False):
        """
        return a list of physical values along a certain dimension, e.g.
        the x-coordinates or timesteps.
        """
        try:
            self.dimensions
        except AttributeError:
            self.get_metadata_dimensions()

        if dim not in self.dimensions or dim == 'channel':
            raise NotImplementedError(f'"{dim}" is not a valid dimension for '+
                                      'visitech_series.get_dimension_steps()')

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
                return (np.arange(0,self.shape[-2]*self._pixelsizeXY,
                                  self._pixelsizeXY),'µm')
            else:
                return (np.arange(0,self.shape[-1]*self._pixelsizeXY,
                                  self._pixelsizeXY),'µm')

        if dim == 'x-axis':
            return (np.arange(0,self.shape[-1]*self._pixelsizeXY,
                              self._pixelsizeXY),'µm')
    
    def get_series_name(self):
        """
        Returns a name for the series based on the filename.

        Returns
        -------
        str
        """
        return self.filename.rpartition('.')[0].rpartition('.')[0]
        
    
    def export_with_scalebar(self,frame=0,filename=None,**kwargs):
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
        resolution : int, optional
            the resolution along the x-axis (i.e. image width in pixels) to use
            for the exported image. The default is `None`, which uses the size 
            of the original image (after optional cropping using `crop`).
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
        cmap_range : tuple of form (min,max) or None or `'automatic'`, optional
            sets the scaling of the colormap. The minimum and maximum 
            values to map the colormap to, values outside of this range will
            be colored according to the min and max value of the colormap. The 
            default is  `None`, which is to take the lowest and highest value 
            in the image. Alternatively `'automatic'` may be specified which 
            scales between the 10th and 99th percentile. For multichannel data
            a list of cmap_range options per channel may be provided.
        draw_bar : boolean, optional
            whether to draw a scalebar on the image, such that this function 
            may be used to put other text on the image or just to apply a 
            colormap (by setting `draw_bar=False` and `draw_text=False`). The 
            default is `True`.
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
        convert : str, one of [`'fm'`,`'pm'`,`'Å'` or `A`,`'nm'`,`'µm'` or `'um'`,`'mm'`,`'cm'`,`'dm'`,`'m'`], optional
            Unit that will be used for the scale bar, the value will be 
            automatically converted if this unit differs from the pixel size
            unit. The default is `None`, which uses micrometers.
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
        draw_text : bool, optional
            whether to draw the text specified in `text` on the image, the text
            is place above the scale bar if `draw_bar=True`. The default is 
            `True`.
        text : str, optional
            the text to draw on the image (above the scale bar if 
            `draw_bar=True`). The default is `None`, which gives the size and 
            unit of the scale bar (e.g. `'10 µm'`).
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
        draw_box : bool, optional
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
        save : bool, optional
            whether to save the image as file. The default is True.
        show_figure : bool, optional
            whether to open matplotlib figure windows. The default is True.
            
        Returns
        -------
        Y×X×4 numpy.array containing the BGRA pixel data
        """       
        #check if pixelsize already calculated, otherwise call get_pixelsize
        pixelsize, unit = self._pixelsizeXY, 'µm'
        
        #set default export filename
        if type(filename) != str:
            filename = self.get_series_name()+'_scalebar.png'
        
        #check we're not overwriting the original file
        if filename==self.filename:
            raise ValueError('overwriting original file not recommended, '+
                             'use a different filename for exporting.')
        
        #get image
        exportim = self.load_data(indices=frame)
        
        #call main export_with_scalebar function with correct pixelsize etc
        from .util import _export_with_scalebar
        return _export_with_scalebar(exportim, pixelsize, unit, filename, 
                                     False, **kwargs)

class visitech_faststack:
    """
    functions for fast stacks taken with the custom MicroManager Visitech 
    driver, saved to multipage .ome.tiff files containing entire stack
    """
    def __init__(self,filename,zsize,zstep,zbacksteps,zstart=0,
                 magnification=63,binning=1):
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
        
        #in case of a multipage ome.tiff
        if isinstance(filename,str):
            self.datafile = pims.TiffStack(filename)
        #in case of list of single images
        elif isinstance(filename,list) or isinstance(filename,np.ndarray):
            self.datafile = pims.ImageSequence(filename)
        else:
            raise ValueError('expected string (for a multipage-tiff) or '
                             'list-like (for image sequence)')
        
        print('PIMS initialized')
        
        #use decimal objects for stack and step size for preventing floating 
        # point errors
        zsize = Decimal('{:}'.format(zsize))
        zstep = Decimal('{:}'.format(zstep))
        
        #find logical sizes of data
        self.nf = len(self.datafile)
        self.nz = int((zsize - (zsize % zstep)) / zstep + 1)
        self.nt = self.nf//(self.nz + zbacksteps)
        self.backsteps = zbacksteps

        #find physical sizes of data
        self.binning = binning
        self.zsteps = np.linspace(
            zstart,
            zstart+float(zsize),
            self.nz,endpoint=True
        )
        self.pixelsize = (
            float(zstep),
            6.5/magnification*binning,
            6.5/magnification*binning
        )
        #Hamamatsu C11440-22CU has pixels of 6.5x6.5 um
        
    def __len__(self):
        """returns length of series, in number of z-stacks"""
        return self.nt
        
    def __repr__(self):
        """"represents class instance in the interpreter"""
        return "<scm_confocal.visitech_series('{}',{},{},{})>".format(
            self.filename,self.zsize,self.zstep,self.backsteps)
    
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

    def load_stack(self,dim_range={},dtype=np.uint16,remove_backsteps=True,
                   offset=0,force_reshape=False):
        """
        Load the data and reshape into 4D stack with the following dimension
        order: ('time','z-axis','y-axis','x-axis')
        
        For loading only part of the total dataset, the dim_range parameter can
        be used to specify a range along any of the dimensions. This will be
        more memory efficient than loading the entire stack and then discarding
        part of the data. For slicing along the `x` or `y` axis this is not
        possible and whole (xy) images must be loaded prior to discarding
        data outside the specified `x` or `y` axis range.
        
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
        force_reshape : bool
            in case of incorrect number of steps during acquisition, you can
            use this to ignore the reshape-error occuring upon trying to sort
            2d images into 4d stack series
            
        Returns
        -------
        data : numpy.ndarray
            ndarray with the pixel values
        """
        #account for offset errors in data recording
        offset = offset % (self.nz+self.backsteps)#offset at most one stack
        
        try: 
            if offset == 0:
                indices = np.reshape(range(self.nf),
                                     (self.nt,self.nz+self.backsteps))
    
            elif offset <= self.backsteps and remove_backsteps:
                #if we remove backsteps and offset is smaller than nbacksteps,
                # we can keep the last stack
                indices = list(range(offset,self.nf))+[0]*offset
                indices = np.reshape(indices,(self.nt,self.nz+self.backsteps))
    
            else:
                #in case of larger offset, lose one stack in total (~half at
                # begin and half at end)
                nf = self.nf - (self.nz+self.backsteps)
                nt = self.nt - 1
                indices = np.reshape(range(offset,offset+nf),
                                     (nt,self.nz+self.backsteps))
        
        #in case the number of images does nog correspond to an integer number
        #of stacks, throw an error or a warning in case of forced loading
        except ValueError:
            if force_reshape:
                print(('[WARNING] scm_confocal.visitech_faststack.load_stack:'+
                       ' cannot reshape {:} images into stack of '+
                      'shape ({:},{:})').format(
                          self.nf,self.nt,self.nz+self.backsteps))
                nt = int(self.nf/(self.nz+self.backsteps))
                nf = nt*(self.nz+self.backsteps)
                
                print('retrying with {:} frames and {:} stacks'.format(nf,nt))
                
                if offset == 0:
                    indices = np.reshape(range(nf),(nt,self.nz+self.backsteps))
            
                elif offset <= self.backsteps and remove_backsteps:
                    #if we remove backsteps and offset is smaller than 
                    # nbacksteps, we can keep the last stack
                    indices = list(range(offset,nf))+[0]*offset
                    indices = np.reshape(indices,(nt,self.nz+self.backsteps))
        
                else:
                    #in case of larger offset, lose one stack in total (~half
                    # at begin and half at end)
                    nf = nf - (self.nz+self.backsteps)
                    nt = nt - 1
                    indices = np.reshape(range(offset,offset+nf),
                                         (nt,self.nz+self.backsteps))
            else:
                raise ValueError(('cannot reshape {:} images into stack of '+
                      'shape ({:},{:})').format(self.nf,self.nt,
                                                self.nz+self.backsteps))

        #remove backsteps from indices
        if remove_backsteps:
            indices = indices[:,:self.nz]
                
        #check dim_range items for faulty values and remove None slices
        for key,val in dim_range.copy().items():
            if type(key) != str or \
                key not in ['time','z-axis','y-axis','x-axis']:
                print("[WARNING] confocal.visitech_faststack.load_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)
            elif val==slice(None):
                dim_range.pop(key)

        #warn for inefficient x and y trimming
        if 'x-axis' in dim_range or 'y-axis' in dim_range:
            print("[WARNING] scm_confocal.visitech_faststack.load_stack: "+
                  "Loading only part of the data along dimensions 'x-axis' "+
                  "and/or 'y-axis' not implemented. Data will be loaded fully"+
                  " into memory before discarding values outside of the "+
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

        #store image indices array for
        # self.get_timestamps(load_stack_indices=True)
        self._stack_indices = indices

        #load and reshape data
        stack = self.load_data(indices=indices.ravel(),dtype=dtype)
        shape = (indices.shape[0],indices.shape[1],stack.shape[1],
                 stack.shape[2])
        stack = stack.reshape(shape)

        #trim x and y axis
        if 'y-axis' in dim_range:
            stack = stack[:,:,dim_range['y-axis']]
        if 'x-axis' in dim_range:
            stack = stack[:,:,:,dim_range['x-axis']]

        return stack

    def yield_stack(self,dim_range={},dtype=np.uint16,remove_backsteps=True,
                    offset=0,force_reshape=False):
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
        force_reshape : bool
            in case of incorrect number of steps during acquisition, you can
            use this to ignore the reshape-error occuring upon trying to sort
            2d images into 4d stack series
            
        Returns
        -------
        zstack : iterable/generator yielding numpy.ndarray
            list of time steps, with for each time step a z-stack as np.ndarray
            with the pixel values
        """
        #account for offset errors in data recording
        offset = offset % (self.nz+self.backsteps)#offset at most one stack

        try: 
            if offset == 0:
                indices = np.reshape(range(self.nf),
                                     (self.nt,self.nz+self.backsteps))
    
            elif offset <= self.backsteps and remove_backsteps:
                #if we remove backsteps and offset is smaller than nbacksteps,
                # we can keep the last stack
                indices = list(range(offset,self.nf))+[0]*offset
                indices = np.reshape(indices,(self.nt,self.nz+self.backsteps))
    
            else:
                #in case of larger offset, lose one stack in total (~half at
                # begin and half at end)
                nf = self.nf - (self.nz+self.backsteps)
                nt = self.nt - 1
                indices = np.reshape(range(offset,offset+nf),
                                     (nt,self.nz+self.backsteps))
        
        #in case the number of images does nog correspond to an integer number
        #of stacks, throw an error or a warning in case of forced loading
        except ValueError:
            if force_reshape:
                print(('[WARNING] scm_confocal.visitech_faststack.yield_stack'+
                       ': cannot reshape {:} images into stack of '+
                      'shape ({:},{:})').format(self.nf,self.nt,
                                                self.nz+self.backsteps))
                nt = int(self.nf/(self.nz+self.backsteps))
                nf = nt*(self.nz+self.backsteps)
                
                print('retrying with {:} frames and {:} stacks'.format(nf,nt))
                
                if offset == 0:
                    indices = np.reshape(range(nf),(nt,self.nz+self.backsteps))
            
                elif offset <= self.backsteps and remove_backsteps:
                    #if we remove backsteps and offset is smaller than
                    # nbacksteps, we can keep the last stack
                    indices = list(range(offset,nf))+[0]*offset
                    indices = np.reshape(indices,(nt,self.nz+self.backsteps))
        
                else:
                    #in case of larger offset, lose one stack in total (~half
                    # at begin and half at end)
                    nf = nf - (self.nz+self.backsteps)
                    nt = nt - 1
                    indices = np.reshape(range(offset,offset+nf),
                                         (nt,self.nz+self.backsteps))
            else:
                raise ValueError(('cannot reshape {:} images into stack of '+
                      'shape ({:},{:})').format(self.nf,self.nt,
                                                self.nz+self.backsteps))

        #remove backsteps from indices
        if remove_backsteps:
            indices = indices[:,:self.nz]

        #check dim_range items for faulty values and remove None slices
        for key,val in dim_range.copy().items():
            if type(key) != str or \
                key not in ['time','z-axis','y-axis','x-axis']:
                print("[WARNING] confocal.visitech_faststack.load_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)
            elif val==slice(None):
                dim_range.pop(key)

        #warn for inefficient x and y trimming
        if 'x-axis' in dim_range or 'y-axis' in dim_range:
            print("[WARNING] scm_confocal.visitech_faststack.yield_stack: "+
                  "Loading only part of the data along dimensions 'x-axis' "+
                  "and/or 'y-axis' not implemented. Data will be loaded fully"+
                  " into memory before discarding values outside of the "+
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

        #store image indices array for
        # self.get_timestamps(load_stack_indices=True)
        self._stack_indices = indices

        #store stack size as attribute
        stack_shape = indices.shape + self.datafile[0].shape
        self.stack_shape = indices.shape + self.datafile[0].shape
        
        if 'y-axis' in dim_range:
            self.stack_shape = self.stack_shape[:2] + (
                len(range(self.stack_shape[2])[dim_range['y-axis']]),
                self.stack_shape[3]
            )
        if 'x-axis' in dim_range:
            self.stack_shape = self.stack_shape[:3] + \
                (len(range(self.stack_shape[3])[dim_range['x-axis']]),)

        #generator loop over each time step in a inner function such that the
        #initialization is excecuted up to this point upon creation rather than
        #upon iteration over the loop
        def stack_iter():
            for zstack_indices in indices:
                zstack = self.load_data(indices=zstack_indices.ravel(),
                                        dtype=dtype)
                zstack = zstack.reshape(stack_shape[1:])
    
                #trim x and y axis
                if 'y-axis' in dim_range:
                    zstack = zstack[:,dim_range['y-axis']]
                if 'x-axis' in dim_range:
                    zstack = zstack[:,:,dim_range['x-axis']]

                yield zstack

        return stack_iter()

    def save_stack(self,data,filename_prefix='visitech_faststack',
                   sequence_type='multipage'):
        """
        save stacks to tiff files
        
        Parameters
        ----------
        data : numpy ndarray with 3 or 4 dimensions
            image series pixel values with dimension order (z,y,x) or (t,z,y,x)
        filename_prefix : string, optional
            prefix to use for filename. The time/z-axis index is appended if
            relevant. The default is 'visitech_faststack'.
        sequence_type : {'multipage','image_sequence','multipage_sequence'}, optional
            The way to store the data. The following options are available:
            
                - 'image_sequence' : stores as a series of 2D images with time 
                and or frame number appended
                - 'multipage' : store all data in a single multipage tiff file
                - 'multipage_sequence' : stores a multipage tiff file for each 
                time step
            
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
                        filename = filename_prefix + \
                            '_t{:03d}_z{:03d}.tif'.format(i,j)
                        Image.fromarray(im).save(filename)
            #for (z,y,x)
            elif len(shape) == 3:
                for i,im in enumerate(data):
                    filename = filename_prefix + '_z{:03d}.tif'.format(i)
                    Image.fromarray(im).save(filename)  
            else:
                raise ValueError('data must be 3-dimensional (z,y,x) or '+
                                 '4-dimensional (t,z,y,x)')
            
        #store as single multipage tiff
        elif sequence_type == 'multipage':
            #for (t,z,y,x)
            if len(shape) == 4:
                data = [Image.fromarray(im) for _ in data for im in _]
                data[0].save(filename_prefix+'.tif',append_images=data[1:],
                             save_all=True,)
            #for (z,y,x)
            elif len(shape) == 3:
                data = [Image.fromarray(im) for im in data]
                data[0].save(filename_prefix+'.tif',append_images=data[1:],
                             save_all=True,)
            else:
                raise ValueError('data must be 3-dimensional (z,y,x) or '+
                                 '4-dimensional (t,z,y,x)')
            
        elif sequence_type == 'multipage_sequence':
            if len(shape) == 4:
                for i,t in enumerate(data):
                    t = [Image.fromarray(im) for im in t]
                    t[0].save(filename_prefix+'_t{:03d}.tif'.format(i),
                              append_images=t[1:],save_all=True)
            elif len(shape) == 3:
                print("[WARNING] scm_confocal.faststack.save_stack(): "+
                      "'multipage_sequence' invalid sequence_type for "+
                      "3-dimensional data. Saving as option 'multipage' "+
                      "instead")
                data = [Image.fromarray(im) for im in data]
                data[0].save(filename_prefix+'.tif',append_images=data[1:],
                             save_all=True)
            else:
                raise ValueError('data must be 4-dimensional (t,z,y,x)')

        else:
            raise ValueError("invalid option for sequence_type: must be "+
                             "'image_sequence', 'multipage' or "+
                             "'multipage_sequence'")
    
    def get_series_name(self):
        """
        Returns a name for the series based on the filename.

        Returns
        -------
        str
        """
        return self.filename.rpartition('.')[0].rpartition('.')[0]    
    
    def _get_metadata_string(self):
        """reads out the raw metadata from a file"""
        import struct
        
        #open file  with bytes
        with open(self.filename,'rb') as file:
            
            #find position of Image File Directory (IDF), given as a 32 bit 
            #unsigned integer in byte position 4 in the file. See TIFF 
            #specification: http://www.exif.org/TIFF6.pdf
            file.seek(4)
            IDF_pos = struct.unpack('I',file.read(4))[0]
            
            #first two bytes are number of entries in IDF, then each entry is 
            #12 bytes long. Per micromanager standard, we need the 5th entry 
            #for OME metadata
            #https://micro-manager.org/wiki/Micro-Manager_Image_File_Stacks
            #bytes 4-7 give item length, bytes 8-11 give byte offset of item as
            #per TIFF standard
            file.seek(IDF_pos+2+5*12+4)
            OME_len = struct.unpack('I',file.read(4))[0]
            OME_offset = struct.unpack('I',file.read(4))[0]
            
            #go to offset and read in the desired number of bytes
            file.seek(OME_offset)
            metadata = file.read(OME_len)

        #decode bytes to string
        metadata = metadata.decode('utf-8')
        
        #cut off extra characters from end if needed
        return metadata[metadata.find('<?xml'):metadata.find('</OME>')+6]

    def _get_metadata_string_imagelist(self):
        """reads out the raw metadata from a list of tiff files"""

        import io
        
        metadata = []
        for file in self.filename[:5]:
            filedat = []
            with io.open(file, 'r', errors='ignore',encoding='utf8') as f:
                for i,line in enumerate(f):
                    if i<=4:
                        continue
                    elif line.replace('\x00','')[:2] != '  ':
                        break
                    filedat.append(line.replace('\x00','')[2:-1])
                metadata.append(filedat)
                    
        
        return metadata

    def get_metadata(self):
        """
        loads OME metadata from visitech .ome.tif file and returns xml tree 
        object
            
        Returns
        -------
        xml.etree.ElementTree
            formatted XML metadata. Can be indexed with
            xml_root.find('<element name>')
        """
        import xml.etree.ElementTree as et

        metadata = self._get_metadata_string()

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
        

        #optionally discard times that were not loaded as part of stack
        if load_stack_indices:
            try:
                indices = self._stack_indices
            except AttributeError:
                raise AttributeError('data must be loaded with '+
                                  '`visitech_faststack.load_stack()` prior to'+
                                  ' calling '+
                                  '`visitech_faststack.get_timestamps()`'+
                                  ' with `load_stack_indices=True`')

        #in case of multipage tiff
        if isinstance(self.filename,str):
            import re
            metadata = self._get_metadata_string()
            times = re.findall(r'DeltaT="([0-9]*\.[0-9]*)"',metadata)
            times = np.array([float(t) for t in times])
            
            if load_stack_indices:
                times = times[indices.ravel()].reshape(np.shape(indices))
        
        #in case of list of tiff images load only the timestamp
        else:    
            if load_stack_indices:
                filenames = self.filename[indices.ravel()]
            else:
                filenames = self.filename
            
            import io
            times = []
            for file in filenames:
                with io.open(file, 'r', errors='ignore',encoding='utf8') as f:
                    for line in f:
                        line = line.replace('\x00','')
                        if '"ElapsedTime-ms"' in line.replace('\x00',''):
                            times.append(line.split(': ')[1][:-2])
                            break
        
            times = np.array([float(t) for t in times])
            
            if load_stack_indices:
                times = times.reshape(np.shape(indices))
            
        self.times = times
        return times

    def get_pixelsize(self):
        """shortcut to get (z,y,x) pixelsize with unit"""
        return (self.pixelsize,'µm')
    
    def export_with_scalebar(self,stack=0,zslice=0,filename=None,**kwargs):
        """
        saves an exported image of the confocal slice with a scalebar in one of
        the four corners, where barsize is the scalebar size in data units 
        (e.g. µm) and scale the overall size of the scalebar and text with 
        respect to the width of the image. Additionally, a colormap is applied
        to the data for better visualisation.

        Parameters
        ----------
        stack : int, optional
            integer index of the z-stack to take the frame to export from. The 
            default is `0`.
        zslice : int, optional
            integer index of the frame within `stack` to export. The default is
            `0`.
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
        resolution : int, optional
            the resolution along the x-axis (i.e. image width in pixels) to use
            for the exported image. The default is `None`, which uses the size 
            of the original image (after optional cropping using `crop`).
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
        cmap_range : tuple of form (min,max) or None or `'automatic'`, optional
            sets the scaling of the colormap. The minimum and maximum 
            values to map the colormap to, values outside of this range will
            be colored according to the min and max value of the colormap. The 
            default is  `None`, which is to take the lowest and highest value 
            in the image. Alternatively `'automatic'` may be specified which 
            scales between the 10th and 99th percentile. For multichannel data
            a list of cmap_range options per channel may be provided.
        draw_bar : boolean, optional
            whether to draw a scalebar on the image, such that this function 
            may be used to put other text on the image or just to apply a 
            colormap (by setting `draw_bar=False` and `draw_text=False`). The 
            default is `True`.
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
        convert : str, one of [`'fm'`,`'pm'`,`'Å'` or `A`,`'nm'`,`'µm'` or `'um'`,`'mm'`,`'cm'`,`'dm'`,`'m'`], optional
            Unit that will be used for the scale bar, the value will be 
            automatically converted if this unit differs from the pixel size
            unit. The default is `None`, which uses micrometers.
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
        draw_text : bool, optional
            whether to draw the text specified in `text` on the image, the text
            is place above the scale bar if `draw_bar=True`. The default is 
            `True`.
        text : str, optional
            the text to draw on the image (above the scale bar if 
            `draw_bar=True`). The default is `None`, which gives the size and 
            unit of the scale bar (e.g. `'10 µm'`).
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
        draw_box : bool, optional
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
        save : bool, optional
            whether to save the image as file. The default is True.
        show_figure : bool, optional
            whether to open matplotlib figure windows. The default is True.
            
        Returns
        -------
        Y×X×4 numpy.array containing the BGRA pixel data
        """    
        #get x pixelsize, unit is always micrometer
        pixelsize, unit = self.pixelsize[2], 'µm'
        
        #set default export filename
        if type(filename) != str:
            filename = self.get_series_name()+'_scalebar.png'
        
        #check we're not overwriting the original file
        if filename==self.filename:
            raise ValueError('overwriting original file not recommended, '+
                             'use a different filename for exporting.')
        
        #check inputs
        if stack >= self.nt:
            raise IndexError(f'stack with index {stack} out of range '
                             f'{self.nt}')
        if zslice >= self.nz:
            raise IndexError(f'zslice with index {zslice} out of range '
                             f'{self.nz}')
        
        #get image
        frame = stack*(self.nz+self.backsteps)+zslice
        exportim = self.load_data(indices=frame)
        
        #call main export_with_scalebar function with correct pixelsize etc
        from .util import _export_with_scalebar
        return _export_with_scalebar(exportim, pixelsize, unit, filename, 
                                     False, **kwargs)
