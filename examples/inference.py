import torch.nn.parallel
import torch
import imageio
import os
import argparse

import soft_renderer.functional as srf
import models

from process_data.mesh_utils import to_unit_edge
import pyopenvdb
from dgts_base import *

MODEL_DIRECTORY = 'inference.tar'
DATASET_DIRECTORY = './data'

SIGMA_VAL = 0.01
IMAGE_SIZE = 64

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)
parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)

# dgts arguments
parser.add_argument('--tag', type=str, help='')
parser.add_argument('--mesh-name', type=str, help='')
parser.add_argument('--template-name', type=str, default='sphere', help='')
parser.add_argument('--num-levels', type=int, help='')
parser.add_argument('--start-level', type=int, default=0, help='')
# inference options
parser.add_argument('--gen-mode', type=str, choices=['generate', 'animate'])
parser.add_argument('--num-gen-samples', type=int, default=8)
parser.add_argument('--target', type=str, default='fertility_al', help='')
parser.add_argument('--gen-levels', nargs='+', type=int, default=[1, 4], help='')
# gt optimization options
parser.add_argument('--template-start', type=int, default=0, help='')

args = parser.parse_args()

directory_output = 'results'
os.makedirs(directory_output, exist_ok=True)

def test(model, image, mesh_path):
    images = torch.from_numpy(image).permute((2,0,1)).unsqueeze(0).cuda()
    vertices, faces = model.reconstruct(images)
    vertices = vertices.squeeze()
    faces = faces.squeeze()
    
    srf.save_obj(mesh_path, vertices, faces)
    return vertices, faces

def run(imagePath, status_bar, progress_bar,textureName = 'sphere_rail',category=''):
   
    model = models.Model('data/sphere/sphere_642.obj', args=args)
    model = model.cuda()

    state_dicts = torch.load(args.model_directory)
    model.load_state_dict(state_dicts['model'], strict=True)
    model.eval()
    status_bar.showMessage('Reconstructing 3d mesh from single image.')
    image = imageio.imread(imagePath, pilmode='RGBA')
    image = image.astype('float32') / 255.
    vertices, faces = test(model, image, f"{directory_output}/tmp.obj")
    status_bar.showMessage('Generating texture on reconstructed 3d mesh.')
    vertices, faces = to_unit_edge((vertices, faces))
    selected_vertices = vertices.cpu().detach().numpy()
    selected_faces = faces.cpu().long().detach().numpy()
    voxel_size = 0.25
    transform = pyopenvdb.createLinearTransform(voxel_size)
    floatGrid = pyopenvdb.FloatGrid.createLevelSetFromPolygons(points=selected_vertices, triangles=selected_faces, transform=transform, halfWidth=1)
    isovalue=0
    adaptivity=0
    vs, triangles, quads = floatGrid.convertToPolygons(isovalue, adaptivity)
    tris = []
    # edge: quad index list
    share_edge_quad_dict = {}
    for idx, quad in enumerate(quads):
        for i in range(4):
            qu = tuple(sorted([quad[i], quad[(i+1)%4]]))
            if qu in share_edge_quad_dict:
                share_edge_quad_dict[qu].append(idx)
            else:
                share_edge_quad_dict[qu] = [idx]

    # quad index: quad index list
    adjacent_quad_dict = {}
    for edge in share_edge_quad_dict:
        quad_list = share_edge_quad_dict[edge]
        for i in range(2):
            if quad_list[i] in adjacent_quad_dict:
                adjacent_quad_dict[quad_list[i]].append(quad_list[(i+1)%2])
            else:
                adjacent_quad_dict[quad_list[i]] = [quad_list[(i+1)%2]]

    ind_queue = [0]
    # quad index: normal direction
    normal_dir_dict = {}
    while True:
        if len(ind_queue) > 0:
            # FIXME Ensure the same face direction
            ind = ind_queue.pop(0)
            if ind not in normal_dir_dict:
                reference_normal = None
                for adjacent_quad_ind in adjacent_quad_dict[ind]:
                    if isinstance(normal_dir_dict.get(adjacent_quad_ind), np.ndarray):
                        reference_normal = normal_dir_dict[adjacent_quad_ind]
                        break
                quad = quads[ind]
                v_0, v_1, v_2 = vs[quad[3]], vs[quad[2]], vs[quad[1]]
                if not isinstance(reference_normal, np.ndarray):
                    normal_dir_dict[ind] = np.cross((v_1 - v_0), (v_1 - v_2))
                    tris.append([quad[3], quad[2], quad[1]])
                    tris.append([quad[3], quad[1], quad[0]])
                else:
                    calculated_normal = np.cross((v_1 - v_0), (v_1 - v_2))
                    if np.dot(calculated_normal, reference_normal) < 0:
                        calculated_normal = -calculated_normal
                        tris.append([quad[3], quad[1], quad[2]])
                        tris.append([quad[3], quad[0], quad[1]])
                    else:
                        tris.append([quad[3], quad[2], quad[1]])
                        tris.append([quad[3], quad[1], quad[0]])
                    normal_dir_dict[ind] = calculated_normal
                    
                # do operation
                for adjacent_quad_ind in adjacent_quad_dict[ind]:
                    if adjacent_quad_ind not in normal_dir_dict:
                        ind_queue.append(adjacent_quad_ind)
        else:
            break
    tris = np.array(tris, dtype=np.int32)
    template = (torch.Tensor(vs).to('cuda'), torch.Tensor(tris).to('cuda').long())
    # Geometric texture synthesis
    # Inference
    opt_ = options.Options()
    opt_.parse_cmdline(parser)
    opt_.gen_mode = 'animate'
    opt_.mesh_name = textureName
    # Wire object output(vertices, faces) to MeshGen and Mesh2Mesh, MeshInference
    # MeshGen done
    # MeshInference done
   
    device = CPU
    with_noise = False
    if opt_.gen_mode == 'generate':
        mg = MeshGen(opt_, device, template)
        mg.generate_all(opt_.num_gen_samples)
    elif opt_.gen_mode == 'animate':
        m2m = Mesh2Mesh(opt_, device)
        
        in_mesh = MeshInference(opt_.target, to_unit_edge(template), opt_, 0, progress_bar=progress_bar).to(device)
        
        output = m2m.animate(in_mesh, opt_.gen_levels[0], opt_.gen_levels[1], 0, (12, 17), zero_places=(0, 0, 1, 1, 1))
        status_bar.showMessage('finish~')
        return output