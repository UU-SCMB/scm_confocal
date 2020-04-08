import numpy as np
import pims
import os
from decimal import Decimal

class visitech_series:
    """
    functions for image series taken with the multi-D acquisition menu in 
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
                indices = np.reshape(range(self.nf),(self.nt,self.nz+self.backsteps))
    
            elif offset <= self.backsteps and remove_backsteps:
                #if we remove backsteps and offset is smaller than nbacksteps, we can keep the last stack
                indices = list(range(offset,self.nf))+[0]*offset
                indices = np.reshape(indices,(self.nt,self.nz+self.backsteps))
    
            else:
                #in case of larger offset, lose one stack in total (~half at begin and half at end)
                nf = self.nf - (self.nz+self.backsteps)
                nt = self.nt - 1
                indices = np.reshape(range(offset,offset+nf),(nt,self.nz+self.backsteps))
        
        #in case the number of images does nog correspond to an integer number
        #of stacks, throw an error or a warning in case of forced loading
        except ValueError:
            if force_reshape:
                print(('[WARNING] scm_confocal.visitech_faststack.load_stack: '+
                       'cannot reshape {:} images into stack of '+
                      'shape ({:},{:})').format(self.nf,self.nt,self.nz+self.backsteps))
                nt = int(self.nf/(self.nz+self.backsteps))
                nf = nt*(self.nz+self.backsteps)
                
                print('retrying with {:} frames and {:} stacks'.format(nf,nt))
                
                if offset == 0:
                    indices = np.reshape(range(nf),(nt,self.nz+self.backsteps))
            
                elif offset <= self.backsteps and remove_backsteps:
                    #if we remove backsteps and offset is smaller than nbacksteps, we can keep the last stack
                    indices = list(range(offset,nf))+[0]*offset
                    indices = np.reshape(indices,(nt,self.nz+self.backsteps))
        
                else:
                    #in case of larger offset, lose one stack in total (~half at begin and half at end)
                    nf = nf - (self.nz+self.backsteps)
                    nt = nt - 1
                    indices = np.reshape(range(offset,offset+nf),(nt,self.nz+self.backsteps))
            else:
                raise ValueError(('cannot reshape {:} images into stack of '+
                      'shape ({:},{:})').format(self.nf,self.nt,self.nz+self.backsteps))

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
            print("[WARNING] scm_confocal.visitech_faststack.load_stack: Loading"+
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
                indices = np.reshape(range(self.nf),(self.nt,self.nz+self.backsteps))
    
            elif offset <= self.backsteps and remove_backsteps:
                #if we remove backsteps and offset is smaller than nbacksteps, we can keep the last stack
                indices = list(range(offset,self.nf))+[0]*offset
                indices = np.reshape(indices,(self.nt,self.nz+self.backsteps))
    
            else:
                #in case of larger offset, lose one stack in total (~half at begin and half at end)
                nf = self.nf - (self.nz+self.backsteps)
                nt = self.nt - 1
                indices = np.reshape(range(offset,offset+nf),(nt,self.nz+self.backsteps))
        
        #in case the number of images does nog correspond to an integer number
        #of stacks, throw an error or a warning in case of forced loading
        except ValueError:
            if force_reshape:
                print(('[WARNING] scm_confocal.visitech_faststack.yield_stack: '+
                       'cannot reshape {:} images into stack of '+
                      'shape ({:},{:})').format(self.nf,self.nt,self.nz+self.backsteps))
                nt = int(self.nf/(self.nz+self.backsteps))
                nf = nt*(self.nz+self.backsteps)
                
                print('retrying with {:} frames and {:} stacks'.format(nf,nt))
                
                if offset == 0:
                    indices = np.reshape(range(nf),(nt,self.nz+self.backsteps))
            
                elif offset <= self.backsteps and remove_backsteps:
                    #if we remove backsteps and offset is smaller than nbacksteps, we can keep the last stack
                    indices = list(range(offset,nf))+[0]*offset
                    indices = np.reshape(indices,(nt,self.nz+self.backsteps))
        
                else:
                    #in case of larger offset, lose one stack in total (~half at begin and half at end)
                    nf = nf - (self.nz+self.backsteps)
                    nt = nt - 1
                    indices = np.reshape(range(offset,offset+nf),(nt,self.nz+self.backsteps))
            else:
                raise ValueError(('cannot reshape {:} images into stack of '+
                      'shape ({:},{:})').format(self.nf,self.nt,self.nz+self.backsteps))

        #remove backsteps from indices
        if remove_backsteps:
            indices = indices[:,:self.nz]

        #check dim_range items for faulty values
        for key in dim_range.keys():
            if type(key) != str or key not in ['time','z-axis','y-axis','x-axis']:
                print("[WARNING] confocal.visitech_faststack.yield_stack: "+
                          "dimension '"+key+"' not present in data, ignoring "+
                          "this entry.")
                dim_range.pop(key)

        #warn for inefficient x and y trimming
        if 'x-axis' in dim_range or 'y-axis' in dim_range:
            print("[WARNING] scm_confocal.visitech_faststack.yield_stack: Loading"+
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
        stack_shape = indices.shape + self.datafile[0].shape
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
                zstack = zstack.reshape(stack_shape[1:])
    
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

        metadata = visitech_faststack._get_metadata_string(self.filename,read_from_end=read_from_end)

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
        imagedata = imagedata.astype({'DeltaT':float,'ExposureTime':float,'PositionZ':float,'TheC':int,'TheT':int,'TheZ':int})

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
        
        #use decimal objects for stack and step size for preventing floating point errors
        zsize = Decimal('{:}'.format(zsize))
        zstep = Decimal('{:}'.format(zstep))
        
        #find logical sizes of data
        self.nf = len(self.datafile)
        self.nz = int((zsize - (zsize % zstep)) / zstep + 1)
        self.nt = self.nf//(self.nz + zbacksteps)
        self.backsteps = zbacksteps

        #find physical sizes of data
        self.binning = binning
        self.zsteps = np.linspace(zstart,zstart+float(zsize),self.nz,endpoint=True)
        self.pixelsize = (float(zstep),6.5/magnification*binning,6.5/magnification*binning)
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
            
        Returns
        -------
        data : numpy.ndarray
            ndarray with the pixel values
        """
        #account for offset errors in data recording
        offset = offset % (self.nz+self.backsteps)#offset at most one stack
        
        if offset == 0:
            indices = np.reshape(range(self.nf),(self.nt,self.nz+self.backsteps))
        
        elif offset <= self.backsteps and remove_backsteps:
            #if we remove backsteps and offset is smaller than nbacksteps, we can keep the last stack
            indices = list(range(offset,self.nf))+[0]*offset
            indices = np.reshape(indices,(self.nt,self.nz+self.backsteps))
        
        else:
            #in case of larger offset, lose one stack in total (~half at begin and half at end)
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
        offset = offset % (self.nz+self.backsteps)#offset at most one stack
        
        if offset == 0:
            indices = np.reshape(range(self.nf),(self.nt,self.nz+self.backsteps))
        
        elif offset <= self.backsteps and remove_backsteps:
            #if we remove backsteps and offset is smaller than nbacksteps, we can keep the last stack
            indices = list(range(offset,self.nf))+[0]*offset
            indices = np.reshape(indices,(self.nt,self.nz+self.backsteps))
        
        else:
            #in case of larger offset, lose one stack in total (~half at begin and half at end)
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
        sequence_type : {'multipage','image_sequence','multipage_sequence'}, optional
            The way to store the data. The following options are available:
            
                - 'image_sequence' : stores as a series of 2D images with time and or frame number appended
                - 'multipage' : store all data in a single multipage tiff file
                - 'multipage_sequence' : stores a multipage tiff file for each time step
            
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
