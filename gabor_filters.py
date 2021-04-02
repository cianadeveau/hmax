def get_gabors(l_sizes, l_divs, n_ori, aspect_ratio):
    """generate the gabor filters

    Args
    ----
        l_sizes: type list of floats
            list of gabor sizes
        l_divs: type list of floats
            list of normalization values to be used
        n_ori: type integer
            number of orientations
        aspect_ratio: type float
            gabor aspect ratio

    Returns
    -------
        gabors: type list of nparrays
            gabor filters

    Example
    -------
        aspect_ratio  = 0.3
        l_gabor_sizes = range(7, 39, 2)
        l_divs        = arange(4, 3.2, -0.05)
        n_ori         = 4
        get_gabors(l_gabor_sizes, l_divs, n_ori, aspect_ratio)

    """

    las = np.array(l_sizes)*2/np.array(l_divs)
    sis = las*0.8
    gabors = []

    # TODO: make the gabors as an array for better handling at the gpu level
    for i, scale in enumerate(l_sizes):
        la = las[i] ; si = sis[i]; gs = l_sizes[i]
        #TODO: inverse the axes in the begining so I don't need to do swap them back
        # thetas for all gabor orientations
        th = np.array(range(n_ori))*np.pi/n_ori + np.pi/2.
        th = th[sp.newaxis, sp.newaxis,:]
        hgs = (gs-1)/2.
        yy, xx = sp.mgrid[-hgs: hgs+1, -hgs: hgs+1]
        xx = xx[:,:,sp.newaxis] ; yy = yy[:,:,sp.newaxis]

        x = xx*np.cos(th) - yy*np.sin(th)
        y = xx*np.sin(th) + yy*np.cos(th)

        filt = np.exp(-(x**2 +(aspect_ratio*y)**2)/(2*si**2))*np.cos(2*np.pi*x/la)
        filt[np.sqrt(x**2+y**2) > gs/2.] = 0

        # gabor normalization (following cns hmaxgray package)
        for ori in range(n_ori):
            filt[:,:,ori] -= filt[:,:,ori].mean()
            filt_norm = fastnorm(filt[:,:,ori])
            if filt_norm !=0: filt[:,:,ori] /= filt_norm
        filt_c = np.array(filt, dtype = 'float32').swapaxes(0,2).swapaxes(1,2)
        gabors.append(filt_c)

    return gabors

def fastnorm(in_arr):
   arr_norm = np.dot(in_arr.ravel(), in_arr.ravel()).sum()**(1./2.)

   return arr_norm

aspect_ratio  = 0.3
l_gabor_sizes = range(7, 39, 2)
l_divs = np.arange(4, 3.2, -0.05)
n_ori = 4
gabors = get_gabors(l_gabor_sizes, l_divs, n_ori, aspect_ratio)
gabor7 = gabors[0:1]
gabor7 = torch.Tensor(gabor7*3)
gabor7 = gabor7.view(4, 3, 7, 7)