import numpy as np

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
        from scipy.optimize import curve_fit
        
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
        (n,A), covariance = curve_fit(f,np.log(x),np.log(y),sigma=weights)
        sigmaN,sigmaA = np.sqrt(np.diag(covariance))
        A = np.exp(A)
        sigmaA = sigmaA*np.exp(A)
        
        return A,n,sigmaA,sigmaN
    
    def mean_square_displacement(features, pos_cols = ['x','y','z'], t_col='t (s)',
                             nparticles=None, pickrandom=False, nbins=20,
                             tmin=None, tmax=None, parallel=False, cores=None):
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
        parallel : bool, optional
            whether to use the parallelized implementation. Requires rest of 
            the code to be protected in a if __name__ == '__main__' block. The
            default is False.
        cores : int, optional
            the number of cores to use when using the parallelized
            implementation. When parallel=False this option is ignored
        
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
        
        #when using parallel processing
        if parallel:
            
            import multiprocessing as mp
            from itertools import repeat
            
            if cores == None:
                cores = mp.cpu_count()
            print('processing MSD using {} cores'.format(cores))
            
            #set up multiprocessing pool and run
            pool = mp.Pool(cores)
            pf = pool.starmap_async(_per_particle_function,zip(particles,repeat(features),repeat(dims)))
            
            #get results and terminate
            pool.close()
            pool.join()
            dt_dr = np.concatenate(pf.get(), axis=0)
            pool.terminate()
            
        
        #normal single core process
        else:
            #initialize empty array to contain [[dt1,dr1],[dt2,dr2],...]
            dt_dr = np.empty((0,2))
            
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
    
    def flatfield_correction_init(images,kernelsize,average=True):
        """
        Provides a correction image for inhomogeneous illumination based on low
        frequency fourier components. Particularly useful for data from the 
        Visitech recorded at relatively large frame size / low imaging rate.

        Parameters
        ----------
        images : (sequence of) numpy array with >= 2 dimensions
            image(s) to calculate a correction image for. The last two 
            dimensions are taken as the 2D images.
        kernelsize : int
            cutoff size in fourier-space pixels (i.e. cycles per image-size) of 
            cone-shaped low-pass fourier filter.
        average : bool, optional
            whether to average correction images along the first dimension of
            the supplied data. Requires >2 dimensions in the input data. The 
            default is True.

        Returns
        -------
        numpy array
            (array of) normalized correction images where the maximum is scaled
            to 1.
        
        See also
        --------
        scm_confocal.util.flatfield_correction_apply
        """
        
        from numpy.fft import rfft2,irfft2,fftshift
        
        #determine shape of fft
        ftshape = (np.shape(images)[-2],np.shape(images)[-1]//2+1)
        
        #create cone kernel in fft-shaped array of zeros
        kernel = np.zeros(ftshape)
        for i in range(-kernelsize-1,kernelsize+1):
            for j in range(kernelsize+2):
                if i**2+j**2<kernelsize**2:
                    kernel[ftshape[0]//2+i,j]=1-(i**2+j**2)/kernelsize**2
        
        #shift kernel along y to match np.fft's default positioning
        kernel = fftshift(kernel,axes=0)
        
        #fourier transform, multiply with kernel, then inverse FT
        corrim = irfft2(rfft2(images)*kernel)
        
        #assume first axis is for averaging
        if average and images.ndim>2:
            corrim = np.mean(corrim,axis=0)
        
        #scale to vals to (0,1] vals and return result
        return corrim/corrim.max()
    
    def flatfield_correction_apply(images,corrim,dtype=None,check_overflow=True):
        """
        Apply a correction to all images in a dataset based on a mask / 
        correction image such as provided by util.flatfield_correction_init.
        Pixel values are divided by the correction image, accounting for 
        integer overflow by clipping to the max value of the (integer) dtype.
        
        Note that overflow checking is currently implemented using numpy masked
        arrays, which are extremely slow (up to 10x) when compared to normal
        numpy arrays. It can be bypassed using check_overflow for a memory and
        performance improvement.

        Parameters
        ----------
        images : (sequence of) numpy.array
            the images to correct. he last two dimensions are taken as the 2D
            images, other dimensions are preserved. Must have 2 or more dims.
        corrim : numpy.array
            The correction image to apply. Must have 2 or more dimensions, if
            there are more than 2 it must match `images` according to numpy
            broadcasting rules.
        dtype : data type, optional
            data type used for the output. The default is images.dtype.
        check_overflow : bool, optional
            Whether to check and avoid integer overflow. The default is True.

        Returns
        -------
        numpy.array
            the corrected image array

        See also
        --------
        scm_confocal.util.flatfield_correction_init
        """
        
        #get data type
        if dtype==None:
            dtype = images.dtype
        
        #bypass overflow checking for a small memory/performance gain
        if not check_overflow:
            (images/corrim).astype(dtype)
        
        #try if integer type, if not it must be float so overflow won't occur
        try:
            maxval = np.iinfo(dtype).max
        except ValueError:
            return (images/corrim).astype(dtype)
        
        #clip overflow values at maxval and correct rest as normal
        mask = images >= maxval*corrim
        images = (np.ma.array(images,mask=mask)/corrim).filled(maxval)
        return images.astype(dtype)
    
    def average_nearest_neighbour_distance(features,pos_cols=['x (um)','y (um)','z (um)']):
        """
        finds average distance of nearest neighbours from pandas array of
        coordinates.

        Parameters
        ----------
        features : pandas DataFrame
            dataframe containing the particle coordinates
        pos_cols : list of strings, optional
            Names of columns to use for coordinates. The default is
            ['x (um)','y (um)','z (um)'].

        Returns
        -------
        float
            average distance to the closest particle for all the pairs in the 
            set

        """
        from scipy.spatial import cKDTree
        
        indices = slice(len(features))
        tree = cKDTree(features[pos_cols])
        pos = tree.data[indices]
        r, i = tree.query(pos,k=2)
        return np.mean(r[:,1])
    
    
    def pair_correlation_3d(features,rmin=0,rmax=10,dr=None,ndensity=None,
                            boundary=None,column_headers=['z','y','x'],
                            periodic_boundary=False,handle_edge=True,):
        """
        calculates g(r) via a 'conventional' distance histogram method for a 
        set of 3D coordinate sets. Edge correction is fully analytic and based 
        on refs [1] and [2].

        Parameters
        ----------
        features : pandas DataFrame or numpy.ndarray
            contains coordinates in (z,y,x)
        rmin : float, optional
            lower bound for the pairwise distance, left edge of 0th bin. The 
            default is 0.
        rmax : float, optional
            upper bound for the pairwise distance, right edge of last bin. The 
            default is 10.
        dr : float, optional
            bin width for the pairwise distance bins. The default is 
            (rmax-rmin)/20.
        ndensity : float, optional
            number density of particles in sample. The default is None which
            computes the number density from the input data.
        boundary : array-like, optional
            positions of the walls that define the bounding box of the 
            coordinates, given as  `(zmin,zmax,ymin,ymax,xmin,xmax)`. The 
            default is the min and max values in the dataset along each 
            dimension.
        column_headers : list of string, optional
            column labels which contain the coordinates to use in case features
            is given as a pandas.DataFrame. The default is ['z','y','x'].
        periodic_boundary : bool, optional
            whether periodic boundary conditions are used. The default is 
            False.
        handle_edge : bool, optional
            whether to correct for edge effects in non-periodic boundary 
            conditions. The default is True.

        Returns
        -------
        edges : numpy.array
            edges of the bins in r
        counts : numpy.array
            normalized count values in each bin of the g(r)

        References
        ----------
        [1] Markus Seserno (2014). How to calculate a three-dimensional g(r) 
        under periodic boundary conditions.
        https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
        
        [2] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
        Distribution Function from Particle Positions: An Advanced Analytic 
        Approach. Analytical Chemistry, 90(23), 13909–13914. 
        https://doi.org/10.1021/acs.analchem.8b03157
        """
        from scipy.spatial import cKDTree
        
        #set default stepsize
        if dr == None:
            dr = (rmax-rmin)/20
        
        #create bin edges and other parameters
        nparticles = len(features)
        edges = np.arange(0,rmax+dr,dr)
        
        #convert to numpy array
        if not isinstance(features,np.ndarray):
            features = features[column_headers].to_numpy()
        
        #set default boundaries to limits of coordinates
        if type(boundary) == type(None):
            xmin, xmax = features[:,2].min(), features[:,2].max()
            ymin, ymax = features[:,1].min(), features[:,1].max()
            zmin, zmax = features[:,0].min(), features[:,0].max()
        
        #otherwise remove particles outside of given limits
        else:
            zmin,zmax,ymin,ymax,xmin,xmax = boundary
            features = features[
                    (features[:,2] >= xmin) & (features[:,2] < xmax) &
                    (features[:,1] >= ymin) & (features[:,1] < ymax) &
                    (features[:,0] >= zmin) & (features[:,0] < zmax)
                    ]
        
        boundary = np.array([[zmin,zmax],[ymin,ymax],[xmin,xmax]])
        
        #calculate number density
        if ndensity == None:
            ndensity = nparticles / np.product(boundary[:,1]-boundary[:,0])
    
        #check rmax and boundary for edge-handling in periodic boundary conditions
        if periodic_boundary:
            if min(boundary[:,1]-boundary[:,0])==max(boundary[:,1]-boundary[:,0]):
                boxlen = boundary[0,1]-boundary[0,0]
                if rmax > boxlen*np.sqrt(3)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(3)/2 times the size of a '+
                        'cubic bounding box when periodic_boundary=True, use '+
                        'rmax < {:}'.format((boundary[0,1]-boundary[0,0])*np.sqrt(3)/2)
                    )
            elif rmax > min(boundary[:,1]-boundary[:,0]):
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '+
                    'periodic_boundary=True is only implemented for cubic boundaries'
                )
        
        #check rmax and boundary for edge handling without periodic boundaries
        else:
            if rmax > max(boundary[:,1]-boundary[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '+
                    'boundary, use rmax < {:}'.format(max(boundary[:,1]-boundary[:,0])/2)
                )
    
        #set up KDTree for fast neighbour finding
        #shift box boundary corner to origin for periodic KDTree
        if periodic_boundary:
            features -= boundary[:,0]
            tree = cKDTree(features-boundary[:,0],boxsize=boundary[:,1]-boundary[:,0])
        else:
            tree = cKDTree(features)
        
        #query tree for any neighbours up to rmax
        dist,indices = tree.query(features,k=nparticles,distance_upper_bound=rmax)
        
        #remove pairs with self, padded (infinite) values and anythin below rmin
        dist = dist[:,1:]
        mask = np.isfinite(dist) & (dist>=rmin)
        
        #when dealing with edges, histogram the distances per reference particle
        #and apply correction factor for missing volume
        if handle_edge:
            if periodic_boundary:
                boundarycorr = _sphere_shell_vol_frac_periodic(
                    edges,
                    min(boundary[:,1]-boundary[:,0])
                )
                counts = np.histogram(dist[mask],bins=edges)/boundarycorr

            else:
                dist = np.ma.masked_array(dist,mask)
                counts = np.apply_along_axis(
                    lambda row: np.histogram(row.data[row.mask],bins=edges)[0],
                    1,
                    dist
                    )
                boundarycorr=_sphere_shell_vol_fraction(
                    edges,
                    boundary-features[:,:,np.newaxis]
                    )
                counts = np.sum(counts/boundarycorr,axis=0)
        
        #otherwise just histogram as a 1d list of distances
        else:
            counts = np.histogram(dist[mask],bins=edges)[0]
        
        #normalize and add to overall list
        counts = counts / (4/3*np.pi * (edges[1:]**3 - edges[:-1]**3)) / (ndensity*nparticles)
        
        return edges,counts
    
    def pair_correlation_2d(features,rmin=0,rmax=10,dr=None,ndensity=None,
                            boundary=None,column_headers=['y','x'],
                            periodic_boundary=False,handle_edge=True,):
        """
        calculates g(r) via a 'conventional' distance histogram method for a 
        set of 2D coordinate sets. Edge correction is fully analytic.

        Parameters
        ----------
        features : pandas DataFrame or numpy.ndarray
            contains coordinates in (y,x)
        rmin : float, optional
            lower bound for the pairwise distance, left edge of 0th bin. The 
            default is 0.
        rmax : float, optional
            upper bound for the pairwise distance, right edge of last bin. The 
            default is 10.
        dr : float, optional
            bin width for the pairwise distance bins. The default is 
            (rmax-rmin)/20.
        ndensity : float, optional
            number density of particles in sample. The default is None which
            computes the number density from the input data.
        boundary : array-like, optional
            positions of the walls that define the bounding box of the 
            coordinates, given as  `(ymin,ymax,xmin,xmax)`. The 
            default is the min and max values in the dataset along each 
            dimension.
        column_headers : list of string, optional
            column labels which contain the coordinates to use in case features
            is given as a pandas.DataFrame. The default is [y','x'].
        periodic_boundary : bool, optional
            whether periodic boundary conditions are used. The default is 
            False.
        handle_edge : bool, optional
            whether to correct for edge effects in non-periodic boundary 
            conditions. The default is True.

        Returns
        -------
        edges : numpy.array
            edges of the bins in r
        counts : numpy.array
            normalized count values in each bin of the g(r)
        """
        from scipy.spatial import cKDTree
        
        #set default stepsize
        if dr == None:
            dr = (rmax-rmin)/20
        
        #create bin edges and other parameters
        nparticles = len(features)
        edges = np.arange(0,rmax+dr,dr)
        
        #convert to numpy array
        if not isinstance(features,np.ndarray):
            features = features[column_headers].to_numpy()
        
        #set default boundaries to limits of coordinates
        if type(boundary) == type(None):
            xmin, xmax = features[:,1].min(), features[:,1].max()
            ymin, ymax = features[:,0].min(), features[:,0].max()
        
        #otherwise remove particles outside of given limits
        else:
            ymin,ymax,xmin,xmax = boundary
            features = features[
                    (features[:,1] >= xmin) & (features[:,1] < xmax) &
                    (features[:,0] >= ymin) & (features[:,0] < ymax)
                    ]
        
        boundary = np.array([[ymin,ymax],[xmin,xmax]])
        
        #calculate number density
        if ndensity == None:
            ndensity = nparticles / np.product(boundary[:,1]-boundary[:,0])
    
        #check rmax and boundary for edge-handling in periodic boundary conditions
        if periodic_boundary:
            if boundary[0,1]-boundary[0,0] == boundary[1,1]-boundary[1,0]:
                boxlen = boundary[0,1]-boundary[0,0]
                if rmax > boxlen*np.sqrt(2)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(2)/2 times the size of a '+
                        'square bounding box when periodic_boundary=True, use '+
                        'rmax < {:}'.format((boundary[0,1]-boundary[0,0])*np.sqrt(2)/2)
                    )
            elif rmax > min(boundary[:,1]-boundary[:,0]):
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '+
                    'periodic_boundary=True is only implemented for square boundaries'
                )
        
        #check rmax and boundary for edge handling without periodic boundaries
        else:
            if rmax > max(boundary[:,1]-boundary[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '+
                    'boundary, use rmax < {:}'.format(max(boundary[:,1]-boundary[:,0])/2)
                )
    
        #set up KDTree for fast neighbour finding
        #shift box boundary corner to origin for periodic KDTree
        if periodic_boundary:
            features -= boundary[:,0]
            tree = cKDTree(features,boxsize=boundary[:,1]-boundary[:,0])
        else:
            tree = cKDTree(features)
        
        #query tree for any neighbours up to rmax
        dist,indices = tree.query(features,k=nparticles,distance_upper_bound=rmax)
        
        #remove pairs with self, padded (infinite) values and anythin below rmin
        dist = dist[:,1:]
        mask = np.isfinite(dist) & (dist>=rmin)
        
        #when dealing with edges, histogram the distances per reference particle
        #and apply correction factor for missing area
        if handle_edge:
            if periodic_boundary:
                boundarycorr = _circle_ring_area_frac_periodic(
                    edges,
                    min(boundary[:,1]-boundary[:,0])
                )
                counts = np.histogram(dist[mask],bins=edges)/boundarycorr

            else:
                dist = np.ma.masked_array(dist,mask)
                counts = np.apply_along_axis(
                    lambda row: np.histogram(row.data[row.mask],bins=edges)[0],
                    1,
                    dist
                    )
                boundarycorr=_circle_ring_area_fraction(
                    edges,
                    boundary-features[:,:,np.newaxis]
                    )
                counts = np.sum(counts/boundarycorr,axis=0)
        
        #otherwise just histogram as a 1d list of distances
        else:
            counts = np.histogram(dist[mask],bins=edges)[0]
        
        #normalize and add to overall list
        counts = counts / (np.pi * (edges[1:]**2 - edges[:-1]**2)) / (ndensity*nparticles)
        
        return edges,counts
    
    def saveprompt(question="Save/overwrite? 1=YES, 0=NO. "):
        """
        Asks user a question (whether to save). If 1 is entered, it returns
        True, for any other answer it returns False
        
        Parameters
        -------
        question : string
            The question to prompt the user for
        
        Returns
        -------
        save : bool
            whether to save
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
        
#define helper function to map over particle list for util.mean_square_displacement
def _per_particle_function(p,features,dims):
    
    #init empty list and particle data
    p_dt_dr = np.empty((0,2))
    pdata = features.loc[p]
    
    #loop over each time interval in particle data and append
    for j in range(len(pdata)):
        for i in range(j):
            p_dt_dr = np.append(
                    p_dt_dr,
                    [[
                            pdata.iat[j,0] - pdata.iat[i,0],
                            sum([(pdata.iat[j,d] - pdata.iat[i,d])**2 for d in range(1,dims+1)])
                    ]],
                    axis = 0
                    )
    return p_dt_dr

#edge correction func for pair_correlation_3d
def _sphere_shell_vol_fraction(r,boundary):
    """fully numpy vectorized function which returns the fraction of the volume 
    of spherical shells r+dr around particles, for a list of particles and a 
    list of bin-edges for the radii simultaneously. Analytical functions for 
    calculating the intersection volume of a sphere and a coboid are taken from
    ref. [1].

    Parameters
    ----------
    r : numpy.array
        list of edges for the bins in r, where the number of shells is len(r)-1
    boundary : numpy.array of shape n*3*2
        list of boundary values shifted with respect to the particle 
        coordinates such that the particles are in the origin, in other words
        the distances to all 6 boundaries. First dimension contains all 
        particles, second dimension refers to spatial dimension (z,y,x) and
        third dimension is used to split the boundaries in the negative and 
        positive directions (or min and max of bounding box in each dimension)

    Returns
    -------
    numpy.array of shape n*(len(r)-1)
        array containing the fraction of each shell in r around each particle
        which lies inside of the boundaries, i.e. v_in/v_tot
    
    References
    ----------
    [1] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
    Distribution Function from Particle Positions: An Advanced Analytic 
    Approach. Analytical Chemistry, 90(23), 13909–13914. 
    https://doi.org/10.1021/acs.analchem.8b03157
    
    See also
    --------
    _sphere_shell_vol_fraction_nb, a numba-compiled version of this function.
    """
    #initialize array with row for each particle and column for each r
    nrow,ncol = len(boundary),len(r)
    vol = np.zeros((nrow,ncol))
    
    #mirror all particle-wall distances into positive octant
    boundary = abs(boundary)
    
    #loop over all sets of three adjecent boundaries
    for hz in (boundary[:,0,0],boundary[:,0,1]):
        for hy in (boundary[:,1,0],boundary[:,1,1]):
            for hx in (boundary[:,2,0],boundary[:,2,1]):
                
                #if box octant entirely inside of sphere octant, add box oct volume
                boxmask = (hx**2+hy**2+hz**2)[:,np.newaxis] < r**2
                vol[boxmask] += np.broadcast_to((hx*hy*hz)[:,np.newaxis],(nrow,ncol))[boxmask]
                
                #to the rest add full sphere octant (or 1/8 sphere)
                boxmask = ~boxmask
                vol[boxmask] += np.broadcast_to((np.pi/6*r**3)[np.newaxis,:],(nrow,ncol))[boxmask]
                
                #remove hemispherical caps
                for h in (hz,hy,hx):
                    
                    #check where to change values, select those items
                    mask = (h[:,np.newaxis] < r)*boxmask
                    indices = np.where(mask)
                    
                    h = h[indices[0]]
                    rs = r[indices[1]]
                    
                    #subtract cap volume
                    vol[mask] -= np.pi/4*(2/3*rs**3-h*rs**2+h**3/3)
                
                #loop over over edges and add back doubly counted edge pieces
                for h0,h1 in ((hz,hy),(hz,hx),(hy,hx)):
                    
                    #check where to change values, select those values
                    mask = (h0[:,np.newaxis]**2+h1[:,np.newaxis]**2 < r**2)*boxmask
                    indices = np.where(mask)

                    h0 = h0[indices[0]]
                    h1 = h1[indices[0]]
                    rs = r[indices[1]]
                    
                    #add back edge wedges
                    c = np.sqrt(rs**2-h0**2-h1**2)
                    vol[indices] += rs**3/6*(np.pi-2*np.arctan(h0*h1/rs/c)) +\
                        (np.arctan(h0/c)-np.pi/2)*(rs**2*h1-h1**3/3)/2 +\
                        (np.arctan(h1/c)-np.pi/2)*(rs**2*h0-h0**3/3)/2 +\
                        h0*h1*c/3
    
    #calculate each shell by subtracting the sphere volumes of previous r
    part_shell = vol[:,1:] - vol[:,:-1]
    tot_shell = 4/3*np.pi * (r[1:]**3 - r[:-1]**3)
    
    return part_shell/tot_shell

def _sphere_shell_vol_frac_periodic(r,boxsize):
    """returns effective volume of each shell defined by the intervals between
    the radii in r, under periodic boundary conditions in a cubic box with 
    edge length boxsize. Effective volume means the volume to which no shorter
    path exists through the periodic boundaries than the r under consideration.
    
    Analytical formulas taken from ref. [1].

    Parameters
    ----------
    r : numpy.array of float
        list of bin edges defining the shells where each interval between 
        values in r is treated as shell.
    boxsize : float
        edge length of the cubic box.

    Returns
    -------
    numpy.array of float
        Effective volume for each interval r -> r+dr in r. Values beyond 
        sqrt(3)/2 boxsize are padded with numpy.nan values.
    
    References
    ----------
    [1] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf

    """
    #init volume list, scale r to boxsize
    vol = np.zeros(len(r))
    r = r/boxsize
    
    #up to half boxlen, normal sphere vol
    mask = r <= 1/2
    vol[mask] = 4/3*np.pi * r[mask]**3
    
    #between boxlen/2 and sqrt(2)/2 boxlen
    mask = (1/2 < r) & (r <= np.sqrt(2)/2)
    vol[mask] = -np.pi/12 * (3 - 36*r[mask]**2 + 32*r[mask]**3)
    
    #between sqrt(2)/2 boxlen and sqrt(3)/2 boxlen
    mask = (np.sqrt(2)/2 < r) & (r <= np.sqrt(3)/2)
    vol[mask] = -np.pi/4 + 3*np.pi*r[mask]**2 + np.sqrt(4*r[mask]**2 - 2) \
        + (1 - 12*r[mask]**2) * np.arctan(np.sqrt(4*r[mask]**2 - 2)) \
        + 2/3 * r[mask]**2 * 8*r[mask] *np.arctan(
            2*r[mask]*(4*r[mask]**2 - 3) / (np.sqrt(4*r[mask]**2 - 2)*(4*r[mask]**2 + 1))
            )
    
    #beyond sqrt(3)/2 boxlen there is no useful info
    vol[np.sqrt(3)/2 < r] = np.nan
    
    part = vol[1:] - vol[:-1]
    full = 4/3*np.pi*(r[1:]**3 - r[:-1]**3)
    
    return part/full

def _circle_ring_area_fraction(r,boundary):
    """fully numpy vectorized function which returns the fraction of the area 
    of circular rings r+dr around particles, for a list of particles and a 
    list of bin-edges for the radii simultaneously. Uses analytical formulas
    for the intersection area of a circle and a rectangle.
    
    Parameters
    ----------
    r : numpy.array
        list of edges for the bins in r, where the number of shells is len(r)-1
    boundary : numpy.array of shape n*2*2
        list of boundary values shifted with respect to the particle 
        coordinates such that the particles are in the origin, in other words
        the distances to all 6 boundaries. First dimension contains all 
        particles, second dimension refers to spatial dimension (y,x) and
        third dimension is used to split the boundaries in the negative and 
        positive directions (or min and max of bounding box in each dimension)

    Returns
    -------
    numpy.array of shape n*(len(r)-1)
        array containing the fraction of each circle ring in r around each 
        particle which lies inside of the boundaries, i.e. A_in/A_tot
    """
    
    #initialize array with row for each particle and column for each r
    nrow,ncol = len(boundary),len(r)
    area = np.zeros((nrow,ncol),dtype=float)
    
    #mirror all particle-edge distances into positive octant
    boundary = abs(boundary)
    
    #loop over all quarters
    for hy in (boundary[:,0,0],boundary[:,0,1]):
        for hx in (boundary[:,1,0],boundary[:,1,1]):
            
            #if circle edge entirely out of boxquarter, add boxquarter area
            boxmask = (hx**2+hy**2)[:,np.newaxis] < r**2
            area[boxmask] += np.broadcast_to((hx*hy)[:,np.newaxis],(nrow,ncol))[boxmask]
            
            #to the rest add a quarter sphere
            boxmask = ~boxmask
            area[boxmask] += np.broadcast_to((np.pi/4*r**2)[np.newaxis,:],(nrow,ncol))[boxmask]
            
            #remove hemispherical caps
            for h in (hy,hx):
                
                #check where to change values, select those items
                mask = (h[:,np.newaxis] < r)*boxmask
                indices = np.where(mask)
                
                h = h[indices[0]]
                rs = r[indices[1]]
                
                #subtract cap area
                area[mask] -= rs*(rs*np.arccos(h/rs) - h*np.sqrt(1-h**2/rs**2))/2
    
    #calculate each shell by subtracting the sphere volumes of previous r
    part_ring = area[:,1:] - area[:,:-1]
    tot_ring = np.pi * (r[1:]**2 - r[:-1]**2)
    
    return part_ring/tot_ring

def _circle_ring_area_frac_periodic(r,boxsize):
    """returns effective area of each ring defined by the intervals between
    the radii in r, under periodic boundary conditions in a square box with 
    edge length boxsize. Effective area means the area to which no shorter
    path exists through the periodic boundaries than the r under consideration.
    
    Parameters
    ----------
    r : numpy.array of float
        list of bin edges defining the shells where each interval between 
        values in r is treated as shell.
    boxsize : float
        edge length of the square box.

    Returns
    -------
    numpy.array of float
        Effective volume for each interval r -> r+dr in r. Values beyond 
        sqrt(2)/2 boxsize are padded with numpy.nan values.

    """
    
    #scale r to boxsize
    r = r/boxsize
    
    #add full circle area to each
    area = np.pi*r**2
    
    #between boxlen/2 and sqrt(2)/2 boxlen subtract 4 circle segments
    mask = (1/2 < r) & (r <= np.sqrt(2)/2)
    area[mask] -= 4*r[mask]*(r[mask]*np.arccos(1/(2*r[mask])) - np.sqrt(1-1/(4*r[mask]**2))/2)
    
    #beyond sqrt(2)/2 boxlen there is no useful info
    area[np.sqrt(2)/2 < r] = np.nan
    
    part_ring = area[1:] - area[:-1]
    full_ring = np.pi*(r[1:]**2 - r[:-1]**2)
    
    return part_ring/full_ring

def _export_with_scalebar(exportim,pixelsize,unit,filename,barsize,crop,scale,
                          loc,resolution,box,invert,convert,cmap,cmap_range):
    """
    see top level export_with_scalebar functions for docs
    """
    #imports
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from PIL import ImageFont, ImageDraw, Image
    import cv2
    
    #get default colormap properties
    if type(cmap) == type(None):
        cmap = 'gray'
    if type(cmap_range) == type(None):
        cmap_range = (np.amin(exportim),np.amax(exportim))
    
    #show original figure
    fig,ax = plt.subplots(1,1)
    ax.imshow(exportim,cmap=cmap,vmin=cmap_range[0],vmax=cmap_range[1])
    plt.title('original image')
    plt.axis('off')
    plt.tight_layout()
    
    #print current axes limits for easy cropping
    def _on_lim_change(call):
        [txt.set_visible(False) for txt in ax.texts]
        xmin,xmax = ax.get_xlim()
        ymax,ymin = ax.get_ylim()
        if len(crop) == 4:
            croptext = 'current crop: (({:}, {:}), ({:}, {:}))'
            croptext = croptext.format(int(xmin),int(ymin),int(xmax+1),int(ymax+1))
        else:
            croptext = 'current crop: ({:}, {:}, {:}, {:})'
            croptext = croptext.format(int(xmin),int(ymin),int(xmax-xmin+1),int(ymax-ymin+1))
        ax.text(0.01,0.01,croptext,fontsize=12,ha='left',va='bottom',
                transform=ax.transAxes,color='red')
    
    #attach callback to limit change
    ax.callbacks.connect("xlim_changed", _on_lim_change)
    ax.callbacks.connect("ylim_changed", _on_lim_change)
    
    #convert unit
    if type(convert) != type(None) and convert != unit:
        
        #always use mu for micrometer
        if convert == 'um':
            convert = 'µm'
        
        #factor 10**3 for every step from list, use indices to calculate
        units = ['pm','nm','µm','mm','m']
        pixelsize = pixelsize*10**(3*(units.index(unit)-units.index(convert)))
    
    #(optionally) crop
    if type(crop) != type(None):
        
        #if (x,y,w,h) format, convert to other format
        if len(crop) == 4:
            crop = ((crop[0],crop[1]),(crop[0]+crop[2],crop[1]+crop[3]))
        
        #crop
        exportim = exportim[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]
        print('cropped to {:} × {:} pixels, {:.4g} × {:.4g} '.format(
            *exportim.shape,exportim.shape[0]*pixelsize,exportim.shape[1]*pixelsize)+unit)
    
    #set default scalebar to original scalebar or calculate len
    if type(barsize) == type(None):
        #take 15% of image width and round to nearest in list of 'nice' vals
        barsize = scale*0.15*exportim.shape[1]*pixelsize
        lst = [0.1,0.2,0.3,0.4,0.5,1,2,2.5,3,4,5,10,20,25,30,
               40,50,100,200,250,300,400,500,1000,2000,2500,
               3000,4000,5000,6000,8000,10000]
        barsize = lst[min(range(len(lst)), key=lambda i: abs(lst[i]-barsize))]
    
    #determine len of scalebar on im
    barsize_px = barsize/pixelsize
    
    #set default resolution or scale image and correct barsize_px
    if type(resolution) == type(None):
        ny,nx = exportim.shape
        resolution = nx
    else:
        nx = resolution
        ny = int(exportim.shape[0]/exportim.shape[1]*nx)
        barsize_px = barsize_px/exportim.shape[1]*resolution
        exportim = cv2.resize(exportim, (int(nx),int(ny)), interpolation=cv2.INTER_AREA)
    
    #rescale to 8 bit interval
    exportim[exportim<cmap_range[0]] = cmap_range[0]
    exportim[exportim>cmap_range[1]] = cmap_range[1]
    exportim  = 255*(exportim - cmap_range[0]) / (cmap_range[1] - cmap_range[0])
    
    #apply colormap
    exportim = cm.get_cmap(cmap,bytes=True)(exportim)
    
    #adjust general scaling for all sizes relative to 1024 pixels
    scale = scale*resolution/1024
    
    #set up sizes
    barheight = scale*16
    boxpad = scale*10
    barpad = scale*10
    textpad = scale*2
    boxalpha = 0.6
    font = 'arialbd.ttf'
    fontsize = 32*scale
    
    #format string
    if round(barsize)==barsize:
        text = str(int(barsize))+' '+unit
    else:
        for i in range(1,4):
            if round(barsize,i)==barsize:
                text = ('{:.'+str(i)+'f} ').format(barsize)+unit
                break
            elif i==3:
                text = '{:.3f} '.format(round(barsize,3))+unit
    
    #get size of text
    #textsize = cv2.getTextSize(text, font, fontsize, int(fontthickness))[0]
    font = ImageFont.truetype(font,size=int(fontsize))
    textsize = ImageDraw.Draw(Image.fromarray(exportim)).textsize(text,font=font)
    offset = font.getoffset(text)
    textsize = (textsize[0]+offset[0],textsize[1]+offset[1])    
    
    #correct baseline for mu in case of micrometer
    if unit=='µm':
        textsize = (textsize[0],textsize[1]-6*scale)
    
    #determine box size
    boxheight = barpad + barheight + 2*textpad + textsize[1]
    
    #determine box position based on loc
    #top left
    if loc == 0:
        x = boxpad
        y = boxpad
    #top right
    elif loc == 1:
        x = nx - boxpad - 2*barpad - max([barsize_px,textsize[0]])
        y = boxpad
    #bottom left
    elif loc == 2:
        x = boxpad
        y = ny - boxpad - boxheight
    #bottom right
    elif loc == 3:
        x = nx - boxpad - 2*barpad - max([barsize_px,textsize[0]])
        y = ny - boxpad - boxheight
    else:
        raise ValueError("loc must be 0, 1, 2 or 3 for top left, top right"+
                         ", bottom left or bottom right respectively.")
    
    #put semitransparent box
    if box:
        #get rectangle from im and create box
        w,h = 2*barpad+max([barsize_px,textsize[0]]),boxheight
        subim = exportim[int(y):int(y+h), int(x):int(x+w)]
        white_box = np.ones(subim.shape, dtype=np.uint8) * 255
        
        #add or subtract box from im, and put back in im
        if invert:
            exportim[int(y):int(y+h), int(x):int(x+w)] = \
                cv2.addWeighted(subim, 1-boxalpha, white_box, boxalpha, 1.0)
        else:
            exportim[int(y):int(y+h), int(x):int(x+w)] = \
                cv2.addWeighted(subim, 1-boxalpha, -white_box, boxalpha, 1.0)

    #calculate positions for bar and text (horizontally centered in box)
    barx = (2*x + 2*barpad + max([barsize_px,textsize[0]]))/2 - barsize_px/2
    bary = y+boxheight-barpad-barheight
    textx = (2*x + 2*barpad + max([barsize_px,textsize[0]]))/2 - textsize[0]/2
    texty = y + textpad
    
    #color for bar and text
    if invert:
        color = 0
    else:
        color = 255
    
    #draw scalebar
    exportim = cv2.rectangle(
        exportim,
        (int(barx),int(bary)),
        (int(barx+barsize_px),int(bary+barheight)),
        color,
        -1
    )
    
    #draw text
    exportim = Image.fromarray(exportim)
    draw = ImageDraw.Draw(exportim)
    draw.text(
        (textx,texty),
        text,
        fill=color,
        font=font
    )
    exportim = np.array(exportim)
    
    #show result
    plt.figure()
    plt.imshow(exportim,cmap='gray',vmin=0,vmax=255)
    plt.title('exported image')
    plt.axis('off')
    plt.tight_layout()
    
    #save image
    cv2.imwrite(filename,exportim)