import argparse

import torch
import numpy as np
from losses import multiview_iou_loss
from utils import AverageMeter, img_cvt
import soft_renderer as sr
import soft_renderer.functional as srf
import datasets
import models
import imageio
import time
import os
import pyopenvdb

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LR_TYPE = 'step'
NUM_ITERATIONS = 250000

LAMBDA_LAPLACIAN = 5e-3
LAMBDA_FLATTEN = 5e-4

PRINT_FREQ = 100
DEMO_FREQ = 1000
SAVE_FREQ = 10000
RANDOM_SEED = 0

MODEL_DIRECTORY = 'data/results/models'
DATASET_DIRECTORY = 'data/datasets'

IMAGE_SIZE = 64
SIGMA_VAL = 1e-4
START_ITERATION = 0

RESUME_PATH = 'data/results/models/recon/checkpoint_0250000.pth.tar'

# SoftRas arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str, default="recon")
parser.add_argument('-md', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-r', '--resume-path', type=str, default=RESUME_PATH)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)
parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE)
parser.add_argument('-lrt', '--lr-type', type=str, default=LR_TYPE)

parser.add_argument('-ll', '--lambda-laplacian', type=float, default=LAMBDA_LAPLACIAN)
parser.add_argument('-lf', '--lambda-flatten', type=float, default=LAMBDA_FLATTEN)
parser.add_argument('-ni', '--num-iterations', type=int, default=NUM_ITERATIONS)
parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
parser.add_argument('-df', '--demo-freq', type=int, default=DEMO_FREQ)
parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)

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

torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

directory_output = os.path.join(args.model_directory, args.experiment_id)
os.makedirs(directory_output, exist_ok=True)
image_output = os.path.join(directory_output, 'pic')
os.makedirs(image_output, exist_ok=True)

# setup model & optimizer
model = models.Model('data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()

optimizer = torch.optim.Adam(model.model_param(), args.learning_rate)

start_iter = START_ITERATION
if args.resume_path:
    state_dicts = torch.load(args.resume_path)
    model.load_state_dict(state_dicts['model'])
    optimizer.load_state_dict(state_dicts['optimizer'])
    start_iter = int(os.path.split(args.resume_path)[1][11:].split('.')[0]) + 1
    print('Resuming from %s iteration' % start_iter)
else:
    dataset_train = datasets.ShapeNet(args.dataset_directory, args.class_ids.split(','), 'train')


def train():
    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for i in range(start_iter, args.num_iterations + 1):
        # adjust learning rate and sigma_val (decay after 150k iter)
        lr = adjust_learning_rate([optimizer], args.learning_rate, i, method=args.lr_type)
        model.set_sigma(adjust_sigma(args.sigma_val, i))

        # load images from multi-view
        images_a, images_b, viewpoints_a, viewpoints_b = dataset_train.get_random_batch(args.batch_size)
        images_a = images_a.cuda()
        images_b = images_b.cuda()
        viewpoints_a = viewpoints_a.cuda()
        viewpoints_b = viewpoints_b.cuda()

        # soft render images
        render_images, laplacian_loss, flatten_loss = model([images_a, images_b],
                                                            [viewpoints_a, viewpoints_b],
                                                            task='train')
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()

        # compute loss
        loss = multiview_iou_loss(render_images, images_a, images_b) + \
            args.lambda_laplacian * laplacian_loss + \
            args.lambda_flatten * flatten_loss
        losses.update(loss.data.item(), images_a.size(0))

        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # save checkpoint
        if i % args.save_freq == 0:
            model_path = os.path.join(directory_output, 'checkpoint_%07d.pth.tar' % i)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, model_path)

        # save demo images
        if i % args.demo_freq == 0:
            demo_image = images_a[0:1]
            demo_path = os.path.join(directory_output, 'demo_%07d.obj' % i)
            demo_v, demo_f = model.reconstruct(demo_image)
            srf.save_obj(demo_path, demo_v[0], demo_f[0])

            imageio.imsave(os.path.join(image_output, '%07d_fake.png' % i), img_cvt(render_images[0][0]))
            imageio.imsave(os.path.join(image_output, '%07d_input.png' % i), img_cvt(images_a[0]))

        # print
        if i % args.print_freq == 0:
            print('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'Loss {loss.val:.3f}\t'
                  'lr {lr:.6f}\t'
                  'sv {sv:.6f}\t'.format(i, args.num_iterations,
                                         batch_time=batch_time, loss=losses,
                                         lr=lr, sv=model.rasterizer.sigma_val))

    return model

def adjust_learning_rate(optimizers, learning_rate, i, method):
    if method == 'step':
        lr, decay = learning_rate, 0.3
        if i >= 150000:
            lr *= decay
    elif method == 'constant':
        lr = learning_rate
    else:
        print("no such learing rate type")

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_sigma(sigma, i):
    decay = 0.3
    if i >= 150000:
        sigma *= decay
    return sigma

if __name__ == '__main__':
    # SoftRas train
    # For debug faster, reuse saved model by args.resumePath
    softRasModel = train()
    # each test image npz is "arr_0": [images[24 angles, 4 channels, 64 height, 64 width]] * n_kind_of_category
    dataset_val = datasets.ShapeNet(args.dataset_directory, args.class_ids.split(','), 'val')
    idx_img = 0
    # 02691156: Airplane, see datasets.py
    class_id = '02691156'
    imgs, vox = next(iter(dataset_val.get_all_batches_for_evaluation(args.batch_size, class_id)))
    imgs = torch.autograd.Variable(imgs).cuda()
    vertices, faces = softRasModel(images=imgs, task='test')
    # Only use one image's output as one batch
    from process_data.mesh_utils import to_unit_edge
    vertices, faces = to_unit_edge((vertices[0], faces[0]))
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
    from dgts_base import *
    opt_ = options.Options()
    opt_.parse_cmdline(parser)
    opt_.gen_mode = 'animate'
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
        in_mesh = MeshInference(opt_.target, to_unit_edge(template), opt_, 0).to(device)
        m2m.animate(in_mesh, opt_.gen_levels[0], opt_.gen_levels[1], 0, (12, 17), zero_places=(0, 0, 1, 1, 1))
