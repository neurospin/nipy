"""
Test the Von-Mises-Fisher mixture model

Author : Bertrand Thirion, 2010 
"""

#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import numpy as np
from numpy.testing import assert_almost_equal

import nipy.neurospin.clustering.von_mises_fisher_mixture as vmm

def test_spherical_area():
    """
    test the co_lavbelling functionality
    """
    points, area = vmm.sphere_density(100)
    assert (np.abs(area.sum()-4*np.pi)<1.e-2)

def test_von_mises_fisher_density():
    """
    test that a density gets indeed computed on the unit sphere
    for a one-component model (k=1)
    """
    x = np.random.randn(100,3)
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    for precision in [.1, 1., 10., 100.]:
        k = 1
        vmd = vmm.VonMisesMixture(k, precision, null_class=False)
        vmd.estimate(x)
    
        # check that it sums to 1
        s, area = vmm.sphere_density(100)
        assert np.abs((vmd.mixture_likelihood(s)*area).sum()-1)<1.e-2

def test_von_mises_fisher_density_plus_null():
    """
    idem test_von_mises_fisher_density, but with a null class
    """
    x = np.random.randn(100,3)
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    for precision in [.1, 1., 10., 100.]:
        k = 1
        vmd = vmm.VonMisesMixture(k, precision, null_class=True)
        vmd.estimate(x)
    
        # check that it sums to 1
        s, area = vmm.sphere_density(100)
        assert np.abs((vmd.mixture_likelihood(s)*area).sum()-1)<1.e-2

def test_von_mises_mixture_density():
    """
    test that a density gets indeed computed on the unit sphere
    for a mixture model (k=3)
    """
    x = np.random.randn(100,3)
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    k = 3
    for precision in [.1, 1., 10., 100.]:
        vmd = vmm.VonMisesMixture(k, precision, null_class=False)
        vmd.estimate(x)
    
        # check that it sums to 1
        s, area = vmm.sphere_density(100)
        assert np.abs((vmd.mixture_likelihood(s)*area).sum()-1)<1.e-2

def test_von_mises_mixture_density_plus_null():
    """
    idem test_von_mises_mixture_density, but with a null class
    """
    x = np.random.randn(100,3)
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    k = 3
    for precision in [.1, 1., 10., 100.]:
        vmd = vmm.VonMisesMixture(k, precision, null_class=True)
        vmd.estimate(x)
    
        # check that it sums to 1
        s, area = vmm.sphere_density(100)
        assert np.abs((vmd.mixture_likelihood(s)*area).sum()-1)<1.e-2



if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


