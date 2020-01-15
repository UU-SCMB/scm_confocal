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
