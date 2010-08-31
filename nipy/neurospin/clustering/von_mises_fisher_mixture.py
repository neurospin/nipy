import numpy as np

class VonMisesMixture(object):
    """
    Model for Von Mises mixture distribution with fixed variance
    on a two-dimensional sphere
    """

    def __init__(self, k, precision, means=None, weights=None,
                 null_class=False):
        """
        Parameters
        ----------
        k: int,
           number of components
        precision: float,
                   the fixed precision parameter
        means: array of shape(self.k, 3), optional
               input component centers
        weights: array of shape(self.k), optional
                 input components weights
        null_class: bool, optional
                    Inclusion of a null class within the model
                    (related to k=0)
        
        fixme
        -----
        consistency checks
        """
        self.k = k
        self.dim = 2
        self.em_dim = 3
        self.means = means
        self.precision = precision
        self.weights = weights
        self.null_class = null_class
        

    def likelihood_per_component(self, x):
        """
        Compute the per-component likelihood of the data

        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        like: array of shape(n, self.k)
        """
        n = x.shape[0]
        constant = self.precision / (2*np.pi*(1-np.exp(-2*self.precision)))
        
        loglike = (np.dot(x, self.means.T)-1)*self.precision
        like = constant*np.exp(loglike)
        if self.null_class:
            like = np.hstack((1./(4*np.pi)*np.ones((n, 1)), like))
        return like

    def weighted_likelihood(self, x):
        """
        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        like: array of shape(n, self.k)
        """
        return(self.likelihood_per_component(x)*self.weights)

    def mixture_likelihood(self, x):
        """
        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        like: array of shape(n)   
        """
        wl = self.weighted_likelihood(x)
        return np.sum(wl, 1)

    def responsibilities(self, x):
        """
        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        resp: array of shape(n, self.k)
        """
        wl = self.weighted_likelihood(x)
        return (wl.T/np.sum(wl, 1)).T

    def estimate_weights(self, z):
        """
        Parameters
        ----------
        z: array of shape(self.k)
        """
        self.weights = np.sum(z,0)/z.sum()

    def estimate_means(self, x, z):
        """
        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere
        z: array of shape(self.k)
        """
        m = np.dot(z.T, x)
        self.means = (m.T/np.sqrt(np.sum(m**2,1))).T

    def estimate(self, x, maxiter=100, miniter=1):
        """
        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere
        maxiter: int, optional,
                 maximum number of iterations of the algorithms
        miniter=1: int, optional,
                 minimum number of iterations
        Return
        ------
        ll: float, average log-likelihood
        """
        # initialization with random positions and constant weights
        if self.weights is None:
            self.weights = np.ones(self.k)/self.k
            if self.null_class:
                self.weights = np.ones(self.k+1)/(self.k+1)
                
        if self.means is None:
            aux = np.arange(x.shape[0])
            np.random.shuffle(aux)
            self.means = x[aux[:self.k]]

        # EM algorithm
        for i in range(maxiter):
            ll = np.log(self.mixture_likelihood(x)).mean()
            z = self.responsibilities(x)
            self.estimate_weights(z)
            if self.null_class:
                self.estimate_means(x, z[:, 1:])
            else:
                self.estimate_means(x, z)
            if i>miniter:
                if ll<pll+1.e-6:
                    break
            pll = ll
        return ll
            

    def show(self, x):
        """
        Visualization utility
        
        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        """
        # label the data
        z = np.argmax(self.responsibilities(x), 1)
        import pylab
        import mpl_toolkits.mplot3d.axes3d as p3
        fig = pylab.figure()
        ax = p3.Axes3D(fig)
        colors = (['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']*(1+(1+self.k)/8))\
                 [:self.k+1]
        if self.null_class:
            ax.plot3D(x[z==0, 0], x[z==0, 1], x[z==0, 2],'.',
                      color=colors[0])
        for k in range(self.k):
            if self.null_class:
                if np.sum(z==k+1)==0:
                    continue
                ax.plot3D(x[z==k+1, 0], x[z==k+1, 1], x[z==k+1, 2],'.',
                          color=colors[k+1])
                ax.plot3D([self.means[k, 0]], [self.means[k, 1]],
                      [self.means[k, 2]], 'o', color=colors[k+1])
            else:
                if np.sum(z==k)==0:
                    continue
                ax.plot3D(x[z==k, 0], x[z==k, 1], x[z==k, 2],'.',
                          color=colors[k])
                ax.plot3D([self.means[k, 0]], [self.means[k, 1]],
                      [self.means[k, 2]], 'o', color=colors[k])
            
        pylab.show()

def estimate_robust_vmm(k, precision, null_class, x, ninit=10, maxiter=100):
    """
    return the best von_mises mixture after severla initialization
    
    Parameters
    ----------
    k: int, number of classes
    precision:
    null class:
    x: array fo shape(n,3)
       should be on the unit sphere
    k: int, optional
    ninit: int, optional,
           number of iterations
    maxiter: int, optional,
             maximum number of iterations after each initialization
    """
    score = -np.infty
    for i in range(ninit):
        aux = VonMisesMixture(k, precision, null_class=null_class)
        ll = aux.estimate(x)
        if ll>score:
            best_model = aux
            score = ll
    return best_model

def select_vmm(krange, precision, null_class, x, ninit=10, maxiter=100):
    """
    return the best von_mises mixture after severla initialization
    
    Parameters
    ----------
    krange: list of ints,
            number of classes to consider
    precision:
    null class:
    x: array fo shape(n,3)
       should be on the unit sphere
    k: int, optional
    ninit: int, optional,
           number of iterations
    maxiter: int, optional,
    """
    score = -np.infty
    for k in krange:
        aux = estimate_robust_vmm(k, precision, null_class, x, ninit, maxiter)
        ll = aux.estimate(x)
        if null_class:
            bic = ll-np.log(x.shape[0])*k*3/x.shape[0]
        else:
             bic = ll-np.log(x.shape[0])*(k*3-1)/x.shape[0]
        if bic>score:
            best_model = aux
            score = bic
    return best_model


def sphere_density(npoints):
    """
    return the points and area of a npoints**2 points sampled on a sphere

    returns
    -------
    s : array of shape(npoints**2, 3)
    area: array of shape(npoints)
    """
    u = np.linspace(0, 2*np.pi, npoints+1)[:npoints]
    v = np.linspace(0, np.pi, npoints+1)[:npoints]
    s = np.vstack((np.ravel(np.outer(np.cos(u), np.sin(v))),
                np.ravel(np.outer(np.sin(u), np.sin(v))),
                   np.ravel(np.outer(np.ones(np.size(u)), np.cos(v))))).T
    area = np.abs(np.ravel(np.outer(np.ones(np.size(u)), np.sin(v))))*\
           np.pi**2*2*1./(npoints**2)
    return s, area

def example():
    x1 = [0.6, 0.48, 0.64]
    x2 = [-0.8, 0.48, 0.36]
    x3 = [0.48, 0.64, -0.6]
    x = np.random.randn(200,3)*.1
    x[:30] += x1
    x[40:150] += x2
    x[150:] += x3
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    precision = 100.
    k = 3
    vmm = select_vmm(range(2,7), precision, True, x)
    #vmm = estimate_robust_vmm(k, precision, True, x)
    #VonMisesMixture(k, precision, null_class=True)
    #vmm.estimate(x)
    vmm.show(x)
    
    # check that it sums to 1
    s, area = sphere_density(100)
    check_integral =  (vmm.mixture_likelihood(s)*area).sum()
