import numpy as	 np
import os.path as op
from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh
import nipy.neurospin.glm_files_layout.tio as tio
from nipy.neurospin.graph.field import field_from_coo_matrix_and_data

# various parameters
mu = 10.
nb_parcel = 150
swd = "/tmp"
laptop = False
contrast_id = ['computation-sentences', 'audio-video', 'left-right', 'reading-visual']
    
for hemisphere in ['left', 'right']:
    if laptop:
        subj_id = 's12069'
        datadir = "/data/thirion/virgile_internship_light/"
        mesh = op.join(datadir,"sphere/ico100_7_lat.tex")
        texfun = [op.join(datadir,"%s/fct/glm/default/Contrast/%s_%s_z_map.tex"%
                          (subj_id, hemisphere, cid)) for cid in contrast_id]
    else:
        datadir = "/neurospin/lnao/Panabase/jirfni2010/Data/DB_SurfaceTest/optimed_localizer/"
        subj_id = 'ag090415'
        mesh = op.join(datadir,'ag090415/t1mri/acq12C/default_analysis/segmentation/mesh/AG090415_%swhite.gii' %hemisphere[0].upper())
        datadir = '/neurospin/tmp/db/optimed_localizer/'
        subj_id = 'AG090415'
        texfun = [op.join(datadir,"%s/fMRI/default_acquisition/cortical_glm/default/Contrast/%s_%s_z_map.tex"%(subj_id, cid, hemisphere)) for cid in contrast_id]

    mesh_dom = domain_from_mesh(mesh, nibabel=False)
    func = np.array([tio.Texture(tf).read(tf).data for tf in texfun]).T

    # prepare the structure
    coord = mesh_dom.coord
    feature = np.hstack((func/func.std(0), mu*coord/coord.std(0)))

    # define the fiels
    df = field_from_coo_matrix_and_data(mesh_dom.topology, feature)
    u, J0 = df.ward(nb_parcel)
    
    # write the result
    tex_labels_name = op.join(swd, "%s_labels.tex" %hemisphere) 
    tio.Texture('', data=u.astype('int32')).write(tex_labels_name)
