
import numpy as	 np
import os.path as op

import parietal.surface_operations.mesh_processing as mep

from gifti import loadImage
# to be replaced with nibabel.gifti

import nipy.neurospin.glm_files_layout.tio as tio

from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import nipy.neurospin.spatial_models.structural_bfls as sbf
from nipy.neurospin.clustering.von_mises_fisher_mixture import select_vmm, VonMisesMixture



def bsa_vmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose=0):
    """
    Estimation of the population level model of activation density using 
    dpmm and inference
    
    Parameters
    ----------
    bf list of nipy.neurospin.spatial_models.hroi.HierarchicalROI instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    gf0, array of shape (nr)
         the mixture-based prior probability 
         that the terminal regions are true positives
    sub, array of shape (nr)
         the subject index associated with the terminal regions
    gfc, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions
    dmax float>0:
         expected cluster std in the common space in units of coord
    thq = 0.5 (float in the [0,1] interval)
        p-value of the prevalence test
    ths=0, float in the rannge [0,nsubj]
        null hypothesis on region prevalence that is rejected during inference
    verbose=0, verbosity mode

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.neurospin.spatial_models.hroi.Nroi instances
        representing individual ROIs
    p: array of shape (nnodes):
       likelihood of the data under H1 over some sampling grid
    """
    dom = bf[0].domain
    n_subj = len(bf)
    
    crmap = -np.ones(dom.size, np.int)
    LR = None
    p = np.zeros(dom.size)
    if len(sub)<1:
        return crmap, LR, bf, p

    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)

    # launch the VMM
    precision = 100.
    vmm = select_vmm(range(5, 20 ), precision, True, gfc)
    if verbose:
        vmm.show(gfc)

    print vmm.k
    z = vmm.responsibilities(gfc)    
    label = np.argmax(vmm.responsibilities(dom.coord), 1)-1
    
    # append some information to the hroi in each subject
    for s in range(n_subj):
        bfs = bf[s]
        if bfs.k>0 :
            leaves = bfs.isleaf()
            us = -np.ones(bfs.k).astype(np.int)

            # set posterior proba
            lq = np.zeros(bfs.k)
            lq[leaves] = 1-z[sub==s, 0]
            bfs.set_roi_feature('posterior_proba', lq)

            # set prior proba
            lq = np.zeros(bfs.k)
            lq[leaves] = 1-gf0[sub==s]
            bfs.set_roi_feature('prior_proba', lq)

            us[leaves] = z[sub==s].argmax(1)-1
            
            # when parent regions has similarly labelled children,
            # include it also
            us = bfs.make_forest().propagate_upward(us)
            bfs.set_roi_feature('label',us)
                        
    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    LR, nl = sbf.build_LR(bf, thq, ths, dmax, verbose=verbose)

    # make a group-level map of the landmark position        
    crmap = bsa._relabel_(label, nl)   
    return crmap, LR, bf, p




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
    r0 = 1.

    mesh_dom = domain_from_mesh(meshes[0])
    ## get the surface-based coordinates
    latitude = tio.Texture(texlat[0]).read(texlat[0]).data
    latitude = latitude-latitude.min()
    longitude = tio.Texture(texlat[0]).read(texlon[0]).data

    #latitude = np.random.rand(mesh_dom.size) * 2  * np.pi
    #longitude = np.random.rand(mesh_dom.size) * np.pi
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
        functional_data = tio.Texture(texfun[s]).read(texfun[s]).data
        #functional_data = np.random.randn(mesh_dom.size)
        
        lbeta.append(functional_data)
        
    lbeta = np.array(lbeta).T
    bf, gf0, sub, gfc = bsa.compute_individual_regions (
        mesh_dom, lbeta, smin, theta, method='prior')

    verbose = 1
    #crmap, LR, bf, p = bsa.bsa_dpmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose)
    crmap, LR, bf, p = bsa_vmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose)
    
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
texfun = [op.join(datadir,"%s/fct/glm/default/Contrast/left_computation-sentences_z_map.tex"%s) for s in subj_id]
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
