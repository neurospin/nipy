
import numpy as	 np
import os.path as op

import nipy.neurospin.glm_files_layout.tio as tio

from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import nipy.neurospin.spatial_models.structural_bfls as sbf
from nipy.neurospin.clustering.von_mises_fisher_mixture import select_vmm_cv
# from nipy.neurospin.clustering.von_mises_fisher_mixture import select_vmm


def kernel_vmm_density(pos, bias, precision, domain):
    """ Define a density on the sphere through a vmm-kde approach
    """
    from nipy.neurospin.clustering.von_mises_fisher_mixture import \
        VonMisesMixture
    from nipy.neurospin.graph.field import field_from_coo_matrix_and_data

    weights = bias / bias.sum()
    k = len(weights)
    tmp = VonMisesMixture(k, 2*precision, means=pos, weights=weights)
    p = tmp.mixture_density(domain.coord)
    sphere_dens = field_from_coo_matrix_and_data(domain.topology, p)
    means = domain.coord[sphere_dens.local_maxima() > 0]
    k = means.shape[0]
    weights = np.ones(k+1) / (k+1)
    vmm = VonMisesMixture(k, precision, means=means, weights=weights, 
                          null_class=True)
    vmm.estimate(pos, bias=bias)
    return vmm


def bsa_vmm(bf, gf0, sub, gfc, dmax, thq, ths, precision=100, verbose=0):
    """ Estimation of the population level model of activation density using
    dpmm and inference

    Parameters
    ----------
    bf list of nipy.neurospin.spatial_models.hroi.HierarchicalROI instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    gf0, array of shape (nr)
         the mixture-based prior probability
         that the terminal regions are false positives
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
    crmap = - np.ones(dom.size, np.int)
    LR = None
    p = np.zeros(dom.size)
    if len(sub) < 1:
        return crmap, LR, bf, p

    sub = np.concatenate(sub).astype(np.int)
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)

    # launch the VMM

    #vmm = select_vmm(range(10, 100, 10 ), precision, gfc, True)
    #vmm = select_vmm_cv(range(10, 50, 5), precision, gfc, null_class=False,
    #                    cv_index=sub, verbose=1)
    #z = vmm.responsibilities(gfc)
    #label = np.argmax(vmm.responsibilities(dom.coord), 1)-1
    #print 'number of components', len(np.unique(label))
    vmm = kernel_vmm_density(gfc, 1-gf0, precision, bf[0].domain)
    
    #vmm = select_vmm_cv(range(5, 50, 5), precision, gfc, null_class=True,
    #                    cv_index=sub, bias=1 - gf0, verbose=1)
    
    if verbose:
        vmm.show(gfc)

    z = vmm.responsibilities(gfc)    
    label = np.argmax(vmm.responsibilities(dom.coord), 1)-1
    print 'number of components', len(np.unique(label))

    p = vmm.mixture_density(dom.coord)

    # append some information to the hroi in each subject
    for s in range(n_subj):
        bfs = bf[s]
        if bfs.k > 0:
            leaves = bfs.isleaf()
            
            # set prior proba
            lq = np.zeros(bfs.k)
            lq[leaves] = 1 - gf0[sub == s]
            bfs.set_roi_feature('prior_proba', lq)
            
            # set posterior proba
            lq = np.zeros(bfs.k)
            lq[leaves] = 1 - z[sub == s, 0]
            bfs.set_roi_feature('posterior_proba', lq)
            
            # when parent regions has similarly labelled children,
            # include it also
            us = - np.ones(bfs.k).astype(np.int)
            us[leaves] = z[sub == s].argmax(1) - 1
            us = bfs.make_forest().propagate_upward(us)
            bfs.set_roi_feature('label', us)
                        
    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    LR, nl = sbf.build_LR(bf, thq, ths, dmax, verbose=1)
    
    # make a group-level map of the landmark position        
    crmap = bsa._relabel_(label, nl).astype(np.int)   
    return crmap, LR, bf, p


def make_surface_BSA(meshes, texfun, texlat, texlon, theta=3.,
                     ths=0, thq=0.5, smin=0, precision=100, swd="/tmp/",
                     contrast_id='cid'):
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
    latitude = latitude - latitude.min()
    longitude = tio.Texture('').read(texlon[0]).data

    coord = r0 * np.vstack((np.sin(latitude) * np.cos(longitude),
                            np.sin(latitude) * np.sin(longitude),
                            np.cos(latitude))).T
    mesh_dom.coord = coord

    lbeta = []
    for s in range(nbsubj):
        # possibly create here subject-specific domain
        #import Texture
        functional_data = tio.Texture(texfun[s]).read(texfun[s]).data
        #functional_data = np.random.randn(mesh_dom.size)
        lbeta.append(functional_data)
        
    lbeta = np.array(lbeta).T
    bf, gf0, sub, gfc = bsa.compute_individual_regions(
        mesh_dom, lbeta, smin, theta, method='prior')
    verbose = 1
    crmap, LR, bf, p = bsa_vmm(
        bf, gf0, sub, gfc, dmax, thq, ths, precision, verbose)
    
    # write the resulting labelling
    tex_labels_name = op.join(swd, "CR_%s.tex" % contrast_id)
    tio.Texture('', data=crmap.astype(np.int32)).write(tex_labels_name)
    
    # write the corresponding density
    tex_labels_name = op.join(swd, "density_%s.tex" % contrast_id) 
    tio.Texture('', data=p).write(tex_labels_name)
    
    # write the prevalence map
    prevalence = np.zeros_like(p)
    prevalence[crmap>-1] =  LR.roi_prevalence()[crmap[crmap>-1]]
    tex_labels_name = op.join(swd, "prevalence_%s.tex" % contrast_id) 
    tio.Texture('', data=prevalence).write(tex_labels_name)
    
    # write the individual maps
    for s in range(nbsubj):
        tex_labels_name = op.join(swd, "AR_s%04d_%s.tex" % (s, contrast_id))
        label = - np.ones(mesh_dom.size, 'int32')
        #
        if bf[s] != None:
            label = bf[s].label.astype('int32')
        tio.Texture('', data=label).write(tex_labels_name)
    return LR, bf
    

theta = 2.5
dmax = 10.
ths = 5
smin = 5
thq = 0.5
precision = 100

subj_id = ['s12069', 's12300', 's12370', 's12405', 's12432', 's12539', 
           's12635', 's12913', 's12081', 's12344', 's12381', 's12414', 
           's12508', 's12562', 's12636', 's12919', 's12165', 's12352', 
           's12401', 's12431', 's12532', 's12590', 's12898', 's12920']
nbsubj = len(subj_id)
datadir = "/data/thirion/virgile_internship_light/"
texlat = [op.join(datadir, "sphere/ico100_7_lat.tex") for s in subj_id]
texlon = [op.join(datadir, "sphere/ico100_7_lon.tex") for s in subj_id]

# left hemisphere
texfun = [op.sep.join((
            datadir, "%s/fct/glm/default/Contrast/" % s,
            "left_computation-sentences_z_map.tex")) for s in subj_id]
meshes = [op.join(datadir, "%s/surf/lh.r.aims.white.gii" % s) for s in subj_id]
#meshes = [op.join(datadir,"sphere/ico100_7.gii") for s in subj_id]
swd = "/tmp"
contrast_id = 'left_computation-sentences'

lr, bf = make_surface_BSA(
    meshes, texfun, texlat, texlon, theta, ths, thq, smin, precision, swd, 
    contrast_id)
