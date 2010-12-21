import numpy as	 np
import os.path as op
import nipy.neurospin.glm_files_layout.tio as tio
from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh
from nipy.neurospin.graph.field import field_from_coo_matrix_and_data

# paths data etc.
subj_id = 's12069'
datadir = "/data/thirion/virgile_internship_light/"
cid = ['computation-sentences','right-left']
swd = "/tmp"
mu = 10.
nb_parcel = 100

for side in ['left', 'right']:
    # data path
    texfun = [op.join(datadir,"%s/fct/glm/default/Contrast/%s_%s_z_map.tex" %
                      (subj_id, side, c)) for c in cid]
    mesh = op.join(datadir,"sphere/ico100_7.gii")
    
    # read the data
    mesh_dom = domain_from_mesh(mesh)
    func = np.array([tio.Texture(t).read(t).data for t in texfun]).T

    # prepare the structure
    coord = mesh_dom.coord
    feature = np.hstack((func/func.std(0), mu*coord/coord.std(0)))

    # define the fiels
    df = field_from_coo_matrix_and_data(mesh_dom.topology, feature)
    u, J0 = df.ward(nb_parcel)

    # write the result
    tex_labels_name = op.join(swd, "%s_labels.tex" %side) 
    tio.Texture('', data=u.astype('int32')).write(tex_labels_name)
