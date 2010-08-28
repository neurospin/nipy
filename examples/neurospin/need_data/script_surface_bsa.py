
import numpy as	 np
import os.path as op

import parietal.surface_operations.mesh_processing as mep

from gifti import loadImage
# to be replaced with nibabel.gifti

import nipy.neurospin.glm_files_layout.tio as tio

from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa



def dpmm(gfc, alpha, g0, g1, dof, prior_precision, gf1, sub, burnin,
         spatial_coords=None, nis=1000, co_clust=False, verbose=False):
    """
    Apply the dpmm analysis to the data: python version
    """
    from nipy.neurospin.clustering.imm import MixedIMM
    dim = gfc.shape[1]
    migmm = MixedIMM(alpha, dim)
    migmm.set_priors(gfc)
    migmm.set_constant_densities(null_dens=g0, prior_dens=g1)
    migmm._prior_dof = dof
    migmm._prior_scale = np.diag(prior_precision[0]/dof)
    migmm._inv_prior_scale_ = [np.diag(dof*1./(prior_precision[0]))]
    migmm.sample(gfc, null_class_proba=1-gf1, niter=burnin, init=False,
                 kfold=sub)
    if verbose:
        print 'number of components: ', migmm.k

    #sampling
    if co_clust:
        like, pproba, co_clust =  migmm.sample(
            gfc, null_class_proba=1-gf1, niter=nis,
            sampling_points=spatial_coords, kfold=sub, co_clustering=co_clust)
        if verbose:
            print 'number of components: ', migmm.k
        
        return like, 1-pproba, co_clust
    else:
        like, pproba =  migmm.sample(
            gfc, null_class_proba=1-gf1, niter=nis,
            sampling_points=spatial_coords, kfold=sub, co_clustering=co_clust)
    if verbose:
        print 'number of components: ', migmm.k
    
    return like, 1-pproba


def make_surface_BSA(meshes, texfun, texlat, texlon, theta=3.,
                     ths = 0, thq = 0.5, smin = 0, swd = "/tmp/",nbeta = [0]):
    """
    Perform the computation of surface landmarks
    this function deals mainly with io

    fixme
    -----
    Write the doc
    replace with nibabel gifti io
    """
    nbsubj = len(meshes)
    coord = []
    r0 = 70.

    mesh_dom = domain_from_mesh(meshes[0])
    ## get the surface-based coordinates
    latitude = tio.Texture(texlat[0]).read(texlat[0]).data
    latitude = latitude-latitude.min()
    longitude = tio.Texture(texlat[0]).read(texlon[0]).data
    print latitude.min(),latitude.max(),longitude.min(),longitude.max()
    latitude = np.random.rand(mesh_dom.size) * 2  * np.pi
    longitude = np.random.rand(mesh_dom.size) * np.pi
    coord = r0*np.vstack((np.sin(latitude) * np.cos(longitude),
                          np.sin(latitude) * np.sin(longitude),
                          np.cos(latitude))).T
    mesh_dom.coord = coord
    
    mesh_doms = []
    lbeta = []
    for s in range(nbsubj):
        
        """
        # this is for subject-specific domains
        mesh_dom = domain_from_mesh(meshes[s])
        
        #import Mesh
        mesh = loadImage(meshes[s])
        vertices = mesh.getArrays()[0].getData()

        ## get the surface-based coordinates
        #latitude = tio.Texture(texlat[s]).read(texlat[s]).data
        #latitude = latitude-latitude.min()
        #longitude = tio.Texture(texlat[s]).read(texlon[s]).data
        #print latitude.min(),latitude.max(),longitude.min(),longitude.max()
        latitude = np.random.rand(vertices.shape[0]) * 2  * np.pi
        longitude = np.random.rand(vertices.shape[0]) * np.pi
        lcoord = r0*np.vstack((np.sin(latitude) * np.cos(longitude),
                               np.sin(latitude) * np.sin(longitude),
                               np.cos(latitude))).T
        
        mesh_dom.coord = lcoord
        mesh_doms.append(mesh_dom)
        """
        
        #import Texture
        #functional_data = tio.Texture(texfun[s][0]).read(texfun[s][0]).data
        functional_data = np.random.randn(mesh_dom.size)
        
        lbeta.append(functional_data)
        
    lbeta = np.array(lbeta).T
    bf, gf0, sub, gfc = bsa.compute_individual_regions (
        mesh_dom, lbeta, smin, theta, method='prior')

    verbose = 1
    crmap, LR, bf, p = bsa.bsa_dpmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose)
    
    """
    v0 = (4*np.pi*r0**2)*np.sqrt(2*np.pi)*dmax
    g0 = 1.0/v0
    bdensity = 1

    
    crmap,AF,BF,p,label = compute_BSA_simple (
    Fs,coord,dmax,thq, smin,ths, theta,g0,bdensity)
    
    
    
    W = aims.Writer()
    if AF!=None:
        defindex = AF.k+2
    else:
        defindex = 0
    
    # write the resulting labelling
    tex_labels_name = op.join(swd,"CR_%04d.tex"%nbeta[0]) 
    nnode = np.size(crmap)
    textureW = aims.TimeTexture_FLOAT()
    tex1 = textureW[0] # First Time sample
    tex1.reserve(nnode)
    for i in range(nnode): tex1.append( crmap[i] )
    W.write( textureW, tex_labels_name)
    
    #write the corresponding density
    tex_labels_name = op.join(swd,"density_%04d.tex"%nbeta[0]) 
    nnode = np.size(crmap)
    textureW = aims.TimeTexture_FLOAT()
    tex1 = textureW[0] # First Time sample
    tex1.reserve(nnode)
    for i in range(nnode): tex1.append( p[i] )
    W.write( textureW, tex_labels_name)
    mesh = R.read(meshes[0])
    print mep.mesh_integrate(mesh,tex1), mep.mesh_integrate(mesh, tex1, coord[0])


    for s in range(nbsubj):
        tex_labels_name = op.join(swd,"AR_s%04d_%04d.tex"%(s,nbeta[0]))
        #nnode = np.size(lw)
        longitudeTex = R.read(texlon[s])
        nnode = np.size(np.array(longitudeTex[0].data()))
        label = -1*np.ones(nnode,'int16')
        #
        if BF[s]!=None:
            nls = BF[s].get_roi_feature('label').copy()
            nls[nls==-1] = defindex
            idx = BF[s].discrete_features['index']
            for k in range(BF[s].k):
                label[idx[k]] =  nls[k]
        #
        textureW = aims.TimeTexture_FLOAT()
        tex1 = textureW[0] # First Time sample
        tex1.reserve(nnode)
        for i in range(nnode): tex1.append( label[i] )
        
        W.write( textureW, tex_labels_name)
    """
    return LR, bf
    


theta = 3.5
dmax = 10.
ths = 2
smin = 5
thq = 0.9

subj_id = [ 's12069',  's12300',  's12370',  's12405',  's12432',  's12539',  's12635',  's12913',  's12081',  's12344',  's12381',  's12414',  's12508',  's12562',  's12636',  's12919',  's12165',  's12352',  's12401',  's12431',  's12532',  's12590',  's12898',  's12920'] [:10]
nbsubj = len(subj_id)
datadir = "/data/thirion/virgile_internship_light/"
texlat = [op.join(datadir,"sphere/ico100_7_lat.tex") for s in subj_id]
texlon = [op.join(datadir,"sphere/ico100_7_lon.tex") for s in subj_id]

# left hemisphere
texfun = [op.join(datadir,"s%s/fct/glm/default/contrast/left_computation-sentences_z_map.tex") for s in subj_id]
#meshes = [op.join(datadir,"s%s/surf/lh.white.gii" %s) for s in subj_id]
meshes = [op.join(datadir,"sphere/ico100_7.gii") for s in subj_id]
swd = "/tmp"

AF,BF = make_surface_BSA(meshes, texfun, texlat, texlon, theta, smin, ths,
                         thq, swd)



"""
subj_id = [ '12069', '12081', '12165', '12207','12300','12344',
             '12352', '12370', '12381', '12401', '12405', '12414',
             '12431', '12432', '12508', '12532', '12539', '12562',
             '12590' ]
nbsubj = len(subj_id)
datadir = "/home/at215559/alanpmad/subjfreesurfer/"
texlat = [op.join(datadir,"ico100_7_lat.tex") for s in subj_id]
texlon = [op.join(datadir,"ico100_7_lon.tex") for s in subj_id]


# left hemisphere
texfun = [[op.join(datadir,"s%s/fct/loc1/L_spmT_%04d.tex") % (s,b) for b in nbeta] for s in subj_id]
meshes = [op.join(datadir,"average_brain/lh.average_brain.mesh") for s in subj_id]
swd = "/tmp/freesurfer/left/"

AF,BF = make_surface_BSA(meshes, texfun,texlat,texlon, theta,smin,ths,thq, dmax, swd,nbeta=nbeta)

# right hemisphere
texfun = [[op.join(datadir,"s%s/fct/loc1/R_spmT_%04d.tex") % (s,b) for b in nbeta] for s in subj_id]
meshes = [op.join(datadir,"average_brain/rh.average_brain.mesh") for s in subj_id]
swd = "/tmp/freesurfer/right/"

AF,BF = make_surface_BSA(meshes, texfun,texlat,texlon, theta,smin,ths,thq, dmax, swd,nbeta=nbeta)
"""




"""
datadir = "/volatile/thirion/Localizer/"
texfun = [[op.join(datadir,"sujet%02d/ftext/method_greg_python/L_spmT_%04d.tex") % (s,b) for b in nbeta] for s in subj_id]

texlat = [op.join(datadir,"sujet%02d/surface/sujet%02d_L_lat.tex") % (s,s) for s in subj_id]
texlon = [op.join(datadir,"sujet%02d/surface/sujet%02d_L_lon.tex") % (s,s) for s in subj_id]

meshes = [op.join(datadir,"sujet%02d/mesh/sujet%02d_Lwhite.mesh") % (s,s) for s in subj_id]
"""
"""
datadir = "/neurospin/lnao/Pmad/alan/jumeaux/Localizer/"
texfun = [[op.join(datadir,"s%s/fMRI/acquisition/loc1/spm_analysis/spm_analysis_Norm_notS/L_spmT_%04d.tex") % (s,b) for b in nbeta] for s in subj_id]
texlat = [op.join(datadir,"s%s/surface/s%s_L_lat.tex") % (s,s) for s in subj_id]
texlon = [op.join(datadir,"s%s/surface/s%s_L_lon.tex") % (s,s) for s in subj_id]
meshes = [op.join(datadir,"s%s/mesh/acquisition/s%s_Lwhite.mesh") % (s,s) for s in subj_id]
swd = "/tmp/left/"
"""
