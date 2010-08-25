
import numpy as	 np
import os.path as op

import parietal.surface_operations.mesh_processing as mep

from gifti import loadImage
import nipy.neurospin.glm_files_layout.tio as tio

from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
from nipy.neurospin.spatial_models.bayesian_structural_analysis import *
from nipy.neurospin.clustering.hierarchical_clustering import average_link_graph, average_link_graph_segment
from nipy.neurospin.spatial_models import hroi
import nipy.neurospin.clustering.clustering as fc
import nipy.neurospin.utils.emp_null as en
import nipy.neurospin.graph.field as ff



def compute_BSA_dev (Fs,Coord,dmax, thq=0.5, smin=5,ths = 0, theta=3.0, g0 = 1.0, bdensity=0,verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns with approach described in IPMI'07 paper
    INPUT:
    - Fs : a list of  fff field classes describing the spatial relationships in the dataset (nbnodes nodes)
    - Beta: an array of size (nbnodes, subjects) with functional data
    - tal: spatial coordinates of the nodes
    - thq = 0.5: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput contains the likelihood of the data under H1 on the set of input nodes
    OUTPUT:
    - crmap: resulting group map
    - AF: list of inter-subject related ROIs
    - BF: List of individual ROIs
    - u: labelling of the individual ROIs
    - p: likelihood of the data under H1 over some sampling grid
    """
    BF = []
    gfc = []
    gf0 = []
    sub = []
    gc = []
    nbsubj = len(Fs)
    header = None

    for s in range(nbsubj):
        Fbeta = Fs[s]
        beta = Fbeta.get_field()
    
        if theta<beta.max():
            idx,height, father,label = Fbeta.threshold_bifurcations(0,theta)
        else:
            idx = []
            father = []
            label = -np.ones(np.shape(beta))
        
        k = np.size(idx)
        nroi = hroi.NROI_from_field(Fbeta,header,Coord[s],refdim=0,th=theta,smin=smin)
        BF.append(nroi)
        # control simple connectivity instead
        if nroi!=None:
            print s, nroi.k
            nvox = np.size(beta)
            sub.append(s*np.ones(nroi.k))
            # find some way to avoid coordinate averaging
            bfm = nroi.discrete_to_roi_features('activation','average')
            nroi.compute_discrete_position()
            bfc = nroi.discrete_to_roi_features('position','cumulated_average')            
            gfc.append(bfc)

            # get some prior on the significance of the regions
            beta = np.reshape(beta,(nvox))
            beta = beta[beta!=0]

            # use a Gamma-Gaussian Mixture Model
            #bfp = bsa._GMM_priors_(beta,bfm,theta,verbose=0)
            bfp = en.three_classes_GMM_fit(beta, bfm, alpha,
                                        prior_strength,verbose,fixed_scale)
            bf0 = bfp[:,1]/np.sum(bfp,1)
            gf0.append(bf0)

            
    tal = Coord[0]
    nnode = tal.shape[0]
    crmap = np.zeros(nnode)     
    if len(sub)<1:
        return None
    
    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    p = np.zeros(np.size(nvox))
    g1 = g0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.float)

    if bdensity:
        spatial_coords = tal
    else:
        spatial_coords = gfc
        
    dof = 0
        
    p,q =  fc.fdp(gfc, 0.5, g0, g1, dof, prior_precision,1-bf0, sub, 100, spatial_coords,10,1000)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')
        mp.show()

    valid = q>thq

    for s in range(nbsubj):
        bfs = BF[s]
        if bfs!=None:
            valids = valid[sub==s]
            valids = bfs.propagate_upward_and(valids)
            bfs.clean(valids)
            bfs.merge_descending()
            
            # re-compute the region position
            bfs.compute_discrete_position()
            bfc = bfs.discrete_to_roi_features('position','cumulated_average')
            
    # compute a model of between-regions associations
    gc = hierarchical_asso(BF,np.sqrt(2)*dmax)

    # Infer the group-level clusters
    if gc == []:
        return crmap,AF,BF,p

    # either replicator dynamics or agglomerative clustering
    #u = sbf.segment_graph_rd(gc,1)
    u,cost = average_link_graph_segment(gc,0.1,gc.V*1.0/nbsubj)

    q = 0
    for s in range(nbsubj):
        if BF[s]!=None:
            BF[s].set_roi_feature('label',u[q:q+BF[s].k])
            q += BF[s].k
    
    LR,mlabel = sbf.build_LR(BF,ths)
    if LR!=None:
        crmap = LR.map_label(tal,pval = 0.95,dmax=dmax)

 
    return crmap,LR,BF,p


def compute_BSA_simple (Fs, Coord, dmax, thq=0.5, smin=5,ths = 0, theta=3.0, g0 = 1.0, bdensity=0,verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns with approach described in IPMI'07 paper
    Parameters
    ----------
    - Fs : a list of  fff field classes describing the spatial relationships in the dataset (nbnodes nodes)
    - Beta: an array of size (nbnodes, subjects) with functional data
    - tal: spatial coordinates of the nodes
    - thq = 0.5: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput contains the likelihood of the data under H1 on the set of input nodes

    Returns
    -------
    - crmap: resulting group map
    - AF: list of inter-subject related ROIs
    - BF: List of individual ROIs
    - u: labelling of the individual ROIs
    - p: likelihood of the data under H1 over some sampling grid
    """
    BF = []
    gfc = []
    gf0 = []
    sub = []
    gc = []
    nbsubj = len(Fs)
    header = None

    for s in range(nbsubj):
        Fbeta = Fs[s]
        beta = Fbeta.get_field()
    
        if theta<beta.max():
            idx,height, father,label = Fbeta.threshold_bifurcations(0,theta)
        else:
            idx = []
            father = []
            label = -np.ones(np.shape(beta))
        
        k = np.size(idx)
        affine = np.eye(4)
        shape = ()
        #nroi = hroi.NROI_from_field(Fbeta, affine, shape, Coord[s], refdim=0,
        # th=theta, smin=smin)
        disc = np.reshape(np.arange(Fbeta.V), (Fbeta.V, 1))
        nroi = hroi.NROI_from_field(Fbeta, affine, shape, disc, refdim=0,
                                    th=theta, smin=smin)
        nroi.set_discrete_feature_from_index('position', Coord[s])
        BF.append(nroi)
        # control simple connectivity instead
        if nroi!=None:
            print s, nroi.k
            nvox = np.size(beta)
            sub.append(s*np.ones(nroi.k))
            # find some way to avoid coordinate averaging
            nroi.set_discrete_feature_from_index('activation',beta)
            bfm = nroi.discrete_to_roi_features('activation','average')
            #nroi.compute_discrete_position()
            bfc = nroi.discrete_to_roi_features('position','cumulated_average')            
            gfc.append(bfc)

            # get some prior on the significance of the regions
            beta = np.reshape(beta,(nvox))
            beta = beta[beta!=0]

            # use a Gamma-Gaussian Mixture Model
            bfp = en.three_classes_GMM_fit(beta, bfm, 0.01,100,verbose,True)
            bf0 = bfp[:,1]/np.sum(bfp,1)
            gf0.append(bf0)

            
    tal = Coord[0]
    nnode = tal.shape[0]
    crmap = np.zeros(nnode)     
    if len(sub)<1:
        return None
    
    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    p = np.zeros(np.size(nvox))
    g1 = g0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.float)
    gf1 = 1-gf0
    
    if bdensity:
        spatial_coords = tal
    else:
        spatial_coords = gfc
        
    dof = 0
    alpha = 0.5
    burnin = 100
    ni_density = 1000
    ni_post = 10
    p,q =  fc.fdp(gfc, alpha, g0, g1, dof, prior_precision,gf1, sub, burnin, spatial_coords,ni_density,ni_post)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')
        mp.show()

    Fbeta = Fs[0]
    Fbeta.set_field(p)
    idx,depth, major,label = Fbeta.custom_watershed(0,g0)

    # append some information to the hroi in each subject
    for s in range(nbsubj):
        bfs = BF[s]
        if bfs!=None:
            leaves = bfs.isleaf()
            us = -np.ones(bfs.k).astype(np.int)
            lq = np.zeros(bfs.k)
            lq[leaves] = q[sub==s]
            bfs.set_roi_feature('posterior_proba',lq)
            lq = np.zeros(bfs.k)
            lq[leaves] = 1-gf0[sub==s]
            bfs.set_roi_feature('prior_proba',lq)
                   
            idx = bfs.feature_argmax('activation')
            #midx = [bfs.discrete_features['index'][k][idx[k]] for k in range(bfs.k)]
            midx = [np.argmin(np.sum(tal-bfs.xyz[k][idx[k]])**2,1) for k in range(bfs.k)]
            j = label[np.array(midx)]
            us[leaves] = j[leaves]

            # when parent regions has similarly labelled children,
            # include it also
            us = bfs.propagate_upward(us)
            bfs.set_roi_feature('label',us)
                        
    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    LR,nl = infer_LR(BF,thq,ths,verbose=verbose)

    # make a group-level map of the landmark position
    crmap = -np.ones(np.shape(label))
    if nl!=None:
        aux = np.arange(label.max()+1)
        aux[0:np.size(nl)]=nl
        crmap[label>-1]=aux[label[label>-1]]
 
    return crmap,LR,BF,p,label

def make_surface_BSA(meshes, texfun, texlat, texlon, theta=3., dmax =  5.,
                     ths = 0, thq = 0.5, smin = 0, swd = "/tmp/",nbeta = [0]):
    """
    Perform the computation of surface landmarks
    this function deals mainly with io

    fixme
    -----
    Write the doc
    replace with nibabel gifti io
    """
    nbsubj = 3#len(meshes)
    coord = []
    r0 = 1.

    mesh_dom = domain_from_mesh(meshes[0])
    ## get the surface-based coordinates
    #latitude = tio.Texture(texlat[s]).read(texlat[s]).data
    #latitude = latitude-latitude.min()
    #longitude = tio.Texture(texlat[s]).read(texlon[s]).data
    #print latitude.min(),latitude.max(),longitude.min(),longitude.max()
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
        mesh_dom, lbeta, smin=10, theta=3.5, method='prior')

    verbose = 1
    crmap, LR, bf, p = bsa_dpmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose)
    
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
    


nbeta = [29] # experimental condition
theta = 2.2
dmax = 5.
ths = 2
smin = 5
thq = 0.9

subj_id = [ '12069']
nbsubj = len(subj_id)
datadir = "/data/thirion/virgile_internship/"
texlat = [op.join(datadir,"s%s/brainvisa_s%s/surface/s%s_L_lat.tex" %(s, s, s))
          for s in subj_id]
texlon = [op.join(datadir,"s%s/brainvisa_s%s/surface/s%s_L_lon.tex" %(s, s, s))
          for s in subj_id]

# left hemisphere
texfun = [[op.join(datadir,"s%s/fct/loc1/L_spmT_%04d.tex") % (s,b) for b in nbeta] for s in subj_id]
meshes = [op.join(datadir,"s%s/surf/lh.white.gii" %s) for s in subj_id]
swd = "/tmp"

AF,BF = make_surface_BSA(meshes, texfun,texlat,texlon, theta,smin,ths,thq, dmax, swd,nbeta=nbeta)



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
