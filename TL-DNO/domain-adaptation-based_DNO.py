import datetime
import os
import random
import time

os.environ["CUDA_VISIBLE_DEVICES"]="8"
from models.FNO import FNO2d, FNO3d
from models.GFNO import GFNO2d, GFNO3d
from models.Unet import UNet2d, UNet3d
# from models.GFNO_steerable import GFNO2d_steer
# from models.Unet import Unet_Rot, Unet_Rot_M, Unet_Rot_3D
from models.Ghybrid import Ghybrid2d
from models.radialNO import radialNO2d, radialNO3d
from models.GCNN import GCNN2d, GCNN3d
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import pandas as pd
from models.DNO_DA import DNO
from torch.utils.data import RandomSampler
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


from utils26 import flood_data, LpLoss, nse, corr, critical_success_index, SinkhornDistance, bce_adv, bce_adv_DS, bce_adv_DT
from models.discriminator1 import FCDiscriminator

import scipy
import numpy as np
from timeit import default_timer
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import h5py
import xarray as xr
from tqdm import tqdm
from openpyxl import load_workbook
torch.set_num_threads(1)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_eval_pred(model, x, strategy, T, times):

    if strategy == "oneshot":
        pred, _ = model(x)
    else:

        for t in range(T):
            t1 = default_timer()
            im, _ = model(x)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)
            if strategy == "markov":
                x = im
            else:
                x = torch.cat((x[..., 1:, :], im), dim=-2)

    return pred

################################################################
# configs
################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--results_path", type=str, default="/Path/to/UrbanFloodCast/Results/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="Flood_DA_DNO_Layers_oneset_seed1_t5", help="suffix to add to the results txt")
parser.add_argument("--super", type=str, default='False', help="enable superres testing")
parser.add_argument("--verbose",type=str, default='True')

parser.add_argument("--T", type=int, default=24, help="number of timesteps to predict")
parser.add_argument("--ntrain", type=int, default=100, help="training sample size")
parser.add_argument("--nvalid", type=int, default=13, help="valid sample size")
parser.add_argument("--ntest", type=int, default=12, help="test sample size")
parser.add_argument("--nsuper", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--model_type", type=str, default='DNO')
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--modes", type=int, default=12)
parser.add_argument("--width", type=int, default=20)
parser.add_argument("--Gwidth", type=int, default=10, help="hidden dimension of equivariant layers if model_type=hybrid")
parser.add_argument("--n_equiv", type=int, default=3, help="number of equivariant layers if model_type=hybrid")
parser.add_argument("--reflection", action="store_true", help="symmetry group p4->p4m for data augmentation")
parser.add_argument("--grid", type=str, default=None, help="[symmetric, cartesian, None]")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stopping", type=int, default=50, help="stop if validation error does not improve for successive epochs")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument('--lr_D', type=float, default=1e-3, metavar='LR_D', help='learning rate (default: auto)')
parser.add_argument("--step", action="store_true", help="use step scheduler")
parser.add_argument("--gamma", type=float, default=0.5, help="gamma for step scheduler")
parser.add_argument("--step_size", type=int, default=None, help="step size for step scheduler")
parser.add_argument("--lmbda", type=float, default=0.0001, help="weight decay for adam")
parser.add_argument("--strategy", type=str, default="oneshot", help="markov, recurrent or oneshot")
parser.add_argument("--time_pad", action="store_true", help="pad the time dimension for strategy=oneshot")
parser.add_argument("--noise_std", type=float, default=0.00, help="amount of noise to inject for strategy=markov")
args = parser.parse_args()

assert args.model_type in ["FNO2d", "FNO2d_aug",
                           "FNO3d", "FNO3d_aug",
                           "UNet2d", "UNet3d", "DNO"], f"Invalid model type {args.model_type}"
assert args.strategy in ["teacher_forcing", "markov", "recurrent", "oneshot"], "Invalid training strategy"

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

data_aug = "aug" in args.model_type

TRAIN_PATH = args.data_path

# FNO data specs
Sy_t = 433
Sx_t = 692
Sy_s = 454
Sx_s = 535
S = 64 # spatial res
S_super = 4 * S # super spatial res
T_in = 1 # number of input times
T = args.T
T_super = 4 * T # prediction temporal super res
d = 2 # spatial res
num_channels = 5
num_channels_y = 3

# adjust data specs based on model type and data path
threeD = args.model_type in ["FNO3d",
                             "Unet_Rot_3D", "DNO", "UNet3d"]
extension = TRAIN_PATH.split(".")[-1]
swe = False
rdb = False
grid_type = "cartesian"
if args.grid:
    grid_type = args.grid
    assert grid_type in ['symmetric', 'cartesian', 'None']


spatial_dims = range(1, d + 1)



ntrain = args.ntrain # 1000
nvalid = args.nvalid
ntest = args.ntest # 200

time_modes = None
time1 = args.strategy == "oneshot" # perform convolutions in space-time
if time1 and not args.time_pad:
    time_modes = 5 if swe else 8 # 6 is based on T=10
elif time1 and swe:
    time_modes = 8

modes = args.modes
width = args.width
n_layer = args.depth
batch_size = args.batch_size

epochs = args.epochs # 500
learning_rate = args.learning_rate
scheduler_step = args.step_size
scheduler_gamma = args.gamma # for step scheduler

initial_step = 1 if args.strategy == "markov" else T_in

root = args.results_path + f"/{'_'.join(str(datetime.datetime.now()).split())}"
if args.suffix:
    root += "_" + args.suffix


path_model = os.path.join(root, 'model_da_adv.pt')
writer = SummaryWriter(root)

# path_model1 = os.path.join(root, 'model.pt')

################################################################
# Model init
################################################################
if args.model_type in ["FNO2d", "FNO2d_aug"]:
    model = FNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                  grid_type=grid_type).cuda()
elif args.model_type in ["FNO3d", "FNO3d_aug"]:
    modes3 = time_modes if time_modes else modes
    model = FNO3d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, modes3=modes3,
                  width=width, time=time1, time_pad=args.time_pad).cuda()
elif args.model_type == "DNO":
    model = DNO(num_channels=num_channels, width=10, initial_step=initial_step, pad=args.time_pad, factor=1).cuda()
elif args.model_type == "UNet3d":
    model = UNet3d(in_channels=initial_step * num_channels, out_channels=num_channels_y, init_features=32,
                   grid_type=grid_type, time=time1).cuda()
else:
    raise NotImplementedError("Model not recognized")

d_h = FCDiscriminator(num_channels=20).cuda()

################################################################
# load data
# Input: DEM/Initial conditions/Rainfall/coords
################################################################
full_data = None # for superres
# Dataset for source domain
dem_tif_path_source = '/Path/to/DEM.tif'
man_path_source = '/Path/to/Strickler_bln1.tif'
Path_train_source = '/Path/to/Source/Train_pt'
Path_valid_source = '/Path/to/Source/Valid_pt'
Path_test_source = '/Path/to/Source/Test_pt'

# Dataset for target domain
dem_tif_path_target = '/Path/to/moa_bottom.tif'
man_path_target = '/Path/to/moa_rough.tif'
Path_train_target_l = '/Path/to/Target/Valid_pt'
Path_train_target = '/Path/to/Target/Train_pt'
Path_train_target_u = '/Path/to/Target/Train_pt'
Path_valid_target = '/Path/to/Target/Test_pt'
Path_test_target = '/Path/to/Target/Test_pt'



train_data_source = flood_data(path_root=Path_train_source, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
ntrain_source = len(train_data_source)
train_data_target_l = flood_data(path_root=Path_train_target_l, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
ntrain_target_l = len(train_data_target_l)
train_data_target_u = flood_data(path_root=Path_train_target_u, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
ntrain_target_u = len(train_data_target_u)
train_data_target = flood_data(path_root=Path_train_target, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
ntrain_target = len(train_data_target)

max_length = max(len(train_data_source), len(train_data_target_l), len(train_data_target_u))
# max_length2 = len(train_data_target)
# def pad_data(data, target_length):
#     while len(data) < target_length:
#         data.append(random.choice(data))
#     return data
# train_data = pad_data(train_data, max_length)
# train_data_u = pad_data(train_data_u, max_length)
print('max_length', max_length)
print('train_data_source', ntrain_source)
print('train_data_target_l', ntrain_target_l)
print('train_data_target_u', ntrain_target_u)


valid_data_target = flood_data(path_root=Path_valid_target, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
nvalid_target = len(valid_data_target)
print('nvalid_target', nvalid_target)

valid_data_source = flood_data(path_root=Path_valid_source, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
nvalid_source = len(valid_data_source)
print('nvalid_source', nvalid_source)

test_data_source = flood_data(path_root=Path_test_source, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
ntest_source = len(test_data_source)
print('ntest_source', ntest_source)

test_data_target = flood_data(path_root=Path_test_target, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
ntest_target = len(test_data_target)
print('ntest_target', ntest_target)

train_loader_source = torch.utils.data.DataLoader(train_data_source, batch_size=batch_size, sampler=RandomSampler(train_data_source, replacement=True, num_samples=max_length))
# train_loader_source2 = torch.utils.data.DataLoader(train_data_source, batch_size=batch_size, sampler=RandomSampler(train_data_source, replacement=True, num_samples=max_length2))
train_loader_target_l = torch.utils.data.DataLoader(train_data_target_l, batch_size=batch_size, sampler=RandomSampler(train_data_target_l, replacement=True, num_samples=max_length))
train_loader_target_u = torch.utils.data.DataLoader(train_data_target_u, batch_size=batch_size, sampler=RandomSampler(train_data_target_u, replacement=True, num_samples=max_length))
# train_loader_target = torch.utils.data.DataLoader(train_data_target, batch_size=batch_size, sampler=RandomSampler(train_data_target, replacement=True, num_samples=max_length2))

valid_loader_source = torch.utils.data.DataLoader(valid_data_source, batch_size=batch_size, shuffle=False)
valid_loader_target = torch.utils.data.DataLoader(valid_data_target, batch_size=batch_size, shuffle=False)
test_loader_source = torch.utils.data.DataLoader(test_data_source, batch_size=batch_size, shuffle=False)
test_loader_target = torch.utils.data.DataLoader(test_data_target, batch_size=batch_size, shuffle=False)


# test_rt_loader = torch.utils.data.DataLoader(test_rt_data, batch_size=batch_size, shuffle=False)
# test_rf_loader = torch.utils.data.DataLoader(test_rf_data, batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
# load the generator (DNO) from pre-trained model
pre_path = '/Path/to/generator/model.pt'
checkpoint = torch.load(pre_path)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
print("Load from checkpoint")

complex_ct = sum(par.numel() * (1 + par.is_complex()) for par in model.parameters())
real_ct = sum(par.numel() for par in model.parameters())
if args.verbose:
    print(f"{args.model_type}; # Params: complex count {complex_ct}, real count: {real_ct}")
writer.add_scalar("Parameters/Complex", complex_ct)
writer.add_scalar("Parameters/Real", real_ct)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.lmbda)
optimizer_d_h = torch.optim.Adam(d_h.parameters(), lr=args.lr_D, betas=(0.9, 0.99))


def poly_lr(iteration):
    max_iterations = epochs * max_length
    return (1 - iteration / max_iterations) ** 0.9

if args.step:
    assert args.step_size is not None, "step_size is None"
    assert scheduler_gamma is not None, "gamma is None"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=scheduler_gamma)
    scheduler_D = LambdaLR(optimizer_d_h, lr_lambda=poly_lr)

else:
    num_training_steps = epochs * max_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
    scheduler_D = LambdaLR(optimizer_d_h, lr_lambda=poly_lr)


# x_num = Sx * Sy
lploss = LpLoss(size_average=False)
sinkloss = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')

# checkpoint = torch.load(path_model1)
# state_dict = checkpoint['state_dict']
# model.load_state_dict(state_dict)

def downsample_maxpool(x):
    pool = nn.MaxPool3d(kernel_size=(x.shape[2] // 16, x.shape[3] // 16, 1))
    return pool(x)

best_valid = float("inf")

model.eval()
start = default_timer()
if args.verbose:
    print("Training...")
step_ct = 0
train_times = []
eval_times = []

source_label = 0
target_label = 1

# Progressive training strategy for stable training
n_critic  = 4

for ep in range(epochs):
    model.train()
    d_h.train()
    t1 = default_timer()

    train_l2 = loss_adv_tl2 = loss_dh_tar_l2 = loss_dh_sou_l2 = loss_adv_t2 = loss_dh_sou2 = loss_dh_tar2 = sink_loss2 = train_vort_l2 = train_pres_l2 = 0

    j = 0
    for (xx_s, yy_s, mask_s), (xx_tl, yy_tl, mask_tl), (xx_tu, yy_tu, mask_tu) in tqdm(zip(train_loader_source, train_loader_target_l, train_loader_target_u), disable=not args.verbose):
        loss = 0
        xx_s = xx_s.cuda()
        xx_tl = xx_tl.cuda()
        xx_tu = xx_tu.cuda()
        yy_s = yy_s.cuda()
        yy_tl = yy_tl.cuda()
        yy_tu = yy_tu.cuda()
        mask_s = mask_s.cuda()
        mask_tl = mask_tl.cuda()
        mask_tu = mask_tu.cuda()
        yy_s = yy_s * mask_s
        yy_tl = yy_tl * mask_tl

        # print('xx_s', xx_s.shape)

        im_s, f_l_s = model(xx_s)
        im_s = im_s * mask_s
        # im_tl, f_l_tl = model(xx_tl)
        # im_tl = im_tl * mask_tl
        im_tu, f_l_tu = model(xx_tu)
        im_tu = im_tu * mask_tu
        # im_s = im_s.permute(0, 4, 1, 2, 3)
        # im_tu = im_tu.permute(0, 4, 1, 2, 3)
        # im_tl = im_tl.permute(0, 4, 1, 2, 3)

        f_l_s = f_l_s.permute(0, 4, 1, 2, 3)
        # f_l_tl = f_l_tl.permute(0, 4, 1, 2, 3)
        f_l_tu = f_l_tu.permute(0, 4, 1, 2, 3)
        # Progressive training strategy for stable training
        if (j + 1) % n_critic == 0:
            optimizer.zero_grad()
            im_tl, f_l_tl = model(xx_tl)
            im_tl = im_tl * mask_tl
            for param in d_h.parameters():
                param.requires_grad = False
            if args.strategy == "oneshot":
                im_s = im_s.squeeze(-1)
                im_tl = im_tl.squeeze(-1)
            loss_s = lploss(im_s.reshape(len(im_s), -1, num_channels_y), yy_s.reshape(len(yy_s), -1, num_channels_y))
            loss_tl = lploss(im_tl.reshape(len(im_tl), -1, num_channels_y), yy_tl.reshape(len(yy_tl), -1, num_channels_y))
            loss = loss_s + loss_tl
            # train on source
            # loss = loss_s
            train_l2 += loss.item()
            loss.backward()

            # adversarial training ot fool the discriminator
            # f_l_tu = f_l_tu.permute(0, 4, 1, 2, 3)
            dh_out_main = d_h(f_l_tu)
            loss_adv_ah = bce_adv(dh_out_main, source_label)
            loss_adv_t = 0.001 * loss_adv_ah
            loss_adv_t2 += loss_adv_t.item()
            loss_adv_t.backward()
            optimizer.step()
            if not args.step:
                scheduler.step()

        # Train discriminator networks
        # enable training mode on discriminator networks
        optimizer_d_h.zero_grad()
        for param in d_h.parameters():
            param.requires_grad = True

        # train witn source
        # domain
        pred_src = f_l_s.detach()

        dh_out_main = d_h(pred_src)
        loss_d_main_sh = bce_adv_DS(dh_out_main, source_label)
        loss_dh_sou = loss_d_main_sh
        loss_dh_sou2 += loss_dh_sou.item()
        loss_dh_sou.backward()

        # train with target
        # domain
        pred_tar = f_l_tu.detach()
        dh_out_main = d_h(pred_tar)
        loss_d_main_th = bce_adv_DT(dh_out_main, target_label)
        loss_dh_tar = loss_d_main_th
        loss_dh_tar2 += loss_dh_tar.item()
        loss_dh_tar.backward()
        optimizer_d_h.step()

        if not args.step:
            scheduler_D.step()
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], step_ct)
        step_ct += 1
        j += 1
        # print('j',j)

    if args.step:
        scheduler.step()
        scheduler_D.step()
        # scheduler2.step()
        # scheduler_D2.step()
        # scheduler_D2.step()

    train_times.append(default_timer() - t1)

    # validation
    valid_l2 = valid_vort_l2 = valid_pres_l2 = 0
    valid_loss_by_channel = None
    with torch.no_grad():
        model.eval()
        for xx, yy, mask in valid_loader_target:

            xx = xx.cuda()
            yy = yy.cuda()
            mask = mask.cuda()
            yy = yy * mask

            # print('xx',xx.shape)
            # print('yy', yy.shape)

            pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T, times=eval_times).view(len(xx), Sy_t, Sx_t, T, num_channels_y)
            # print('pred',pred.shape)
            # print('yy', yy.shape)
            pred = pred * mask

            valid_l2 += lploss(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()

    t2 = default_timer()
    if args.verbose:
        print(f"Ep: {ep}, time: {t2 - t1}, train: {train_l2 / ntrain}, valid_t: {valid_l2 / nvalid}, loss_adv_t: {loss_adv_t2 / ntrain}, loss_dh_sou: {loss_dh_sou2 / ntrain}, loss_dh_tar: {loss_dh_tar2 / ntrain}")

    writer.add_scalar("Train/Loss", train_l2 / ntrain, ep)
    writer.add_scalar("Valid/Loss", valid_l2 / nvalid, ep)

    if valid_l2 < best_valid:
        best_epoch = ep
        best_valid = valid_l2
        # torch.save(model.state_dict(), path_model)
        state_dict = model.state_dict()
        torch.save({'epoch': best_epoch, 'state_dict': state_dict}, path_model)
    if args.early_stopping:
        if ep - best_epoch > args.early_stopping:
            break
def generate_movie_2D(key, test_x, test_y, preds_y, plot_title='', field=0, val_cbar_index=-1, err_cbar_index=-1,
                      val_clim=None, err_clim=None, font_size=None, movie_dir='', movie_name='movie.gif',
                      frame_basename='movie', frame_ext='jpg', remove_frames=True):
    frame_files = []

    if movie_dir:
        os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    if len(preds_y.shape) == 4:
        Nsamples, Nx, Ny, Nt = preds_y.shape
        preds_y = preds_y.reshape(Nsamples, Nx, Ny, Nt, 1)
        test_y = test_y.reshape(Nsamples, Nx, Ny, Nt, 1)
    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape
    print('preds_y', preds_y.shape)

    pred = preds_y[key, ..., field]
    true = test_y[key, ..., field]
    error = torch.abs(pred - true)

    a = test_x[key]
    x = torch.linspace(0, 1, Nx + 1)[:-1]
    y = torch.linspace(0, 1, Ny + 1)[:-1]
    X, Y = torch.meshgrid(x, y)
    # t = a[0, 0, :, 2]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    colors = plt.cm.viridis(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    cmap = ListedColormap(colors)

    pcm1 = ax1.pcolormesh(X, Y, true[..., val_cbar_index], cmap=cmap, label='true', shading='gouraud')
    pcm2 = ax2.pcolormesh(X, Y, pred[..., val_cbar_index], cmap=cmap, label='pred', shading='gouraud')
    pcm3 = ax3.pcolormesh(X, Y, error[..., err_cbar_index], cmap=cmap, label='error', shading='gouraud')

    if val_clim is None:
        val_clim = pcm1.get_clim()
    if err_clim is None:
        err_clim = pcm3.get_clim()

    pcm1.set_clim(val_clim)
    plt.colorbar(pcm1, ax=ax1)
    ax1.axis('square')

    pcm2.set_clim(val_clim)
    plt.colorbar(pcm2, ax=ax2)
    ax2.axis('square')

    pcm3.set_clim(err_clim)
    plt.colorbar(pcm3, ax=ax3)
    ax3.axis('square')

    plt.tight_layout()

    for i in range(Nt):
        # Exact
        ax1.clear()
        pcm1 = ax1.pcolormesh(X, Y, true[..., i], cmap=cmap, label='true', shading='gouraud')
        pcm1.set_clim(val_clim)
        ax1.set_title(f'Hydraulic Model {plot_title}')
        ax1.axis('square')

        # Predictions
        ax2.clear()
        pcm2 = ax2.pcolormesh(X, Y, pred[..., i], cmap=cmap, label='pred', shading='gouraud')
        pcm2.set_clim(val_clim)
        ax2.set_title(f'KI-Tool {plot_title}')
        ax2.axis('square')

        # Error
        ax3.clear()
        pcm3 = ax3.pcolormesh(X, Y, error[..., i], cmap=cmap, label='error', shading='gouraud')
        pcm3.set_clim(err_clim)
        ax3.set_title(f'Error {plot_title}')
        ax3.axis('square')

        #         plt.tight_layout()
        fig.canvas.draw()

        if movie_dir:
            frame_path = os.path.join(movie_dir, f'{frame_basename}-{i:03}.{frame_ext}')
            frame_files.append(frame_path)
            plt.savefig(frame_path)

    if movie_dir:
        movie_path = os.path.join(movie_dir, movie_name)
        with imageio.get_writer(movie_path, mode='I') as writer:
            for frame in frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

    # if movie_dir and remove_frames:
    #     for frame in frame_files:
    #         try:
    #             os.remove(frame)
    #         except:
    #             pass
stop = default_timer()
train_time = stop - start
train_times = torch.tensor(train_times).mean().item()
num_eval = len(eval_times)
eval_times = torch.tensor(eval_times).mean().item()
model.eval()
# test
##FNO
# model.load_state_dict(torch.load(path_model))
## Other models
checkpoint = torch.load(path_model)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.eval()
test_l2_s = test_vort_l2 = test_pres_l2 = test_nse_s = test_corr_s = test_csi_1_s = test_csi_2_s = test_csi_3_s = 0
test_l2_t = test_nse_t = test_corr_t = test_csi_1_t = test_csi_2_t = test_csi_3_t = 0
rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None

total_time_s = 0
sample_count_s = 0
total_time_t = 0
sample_count_t = 0

with torch.no_grad():
    for xx, yy, mask in test_loader_source:
        xx = xx.cuda()
        yy = yy.cuda()
        mask = mask.cuda()
        yy = yy * mask
        input_data = xx
        # print('xx', xx.shape)
        # print('yy', yy.shape)
        # Start
        start_time = time.time()
        pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T, times=[]).view(len(xx), Sy_s, Sx_s, T, num_channels_y)
        # End
        end_time = time.time()
        batch_time = end_time - start_time
        total_time_s += batch_time
        sample_count_s += len(xx)
        # print(f"Average prediction time per sample: {batch_time:.4f} seconds, lens of samples: {len(xx)}")

        pred = pred * mask
        # print('pred', pred.shape)
        test_l2_s += lploss(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_nse_s += nse(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_corr_s += corr(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_csi_1_s += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.01).item()
        test_csi_2_s += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.1).item()
        test_csi_3_s += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.5).item()

with torch.no_grad():
    for xx, yy, mask in test_loader_target:
        xx = xx.cuda()
        yy = yy.cuda()
        mask = mask.cuda()
        yy = yy * mask
        input_data = xx
        # print('xx', xx.shape)
        # print('yy', yy.shape)
        # Start
        start_time = time.time()
        pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T, times=[]).view(len(xx), Sy_t, Sx_t, T, num_channels_y)
        # End
        end_time = time.time()
        batch_time = end_time - start_time
        total_time_t += batch_time
        sample_count_t += len(xx)
        # print(f"Average prediction time per sample: {batch_time:.4f} seconds, lens of samples: {len(xx)}")

        pred = pred * mask
        # print('pred', pred.shape)
        test_l2_t += lploss(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_nse_t += nse(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_corr_t += corr(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_csi_1_t += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.01).item()
        test_csi_2_t += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.1).item()
        test_csi_3_t += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.5).item()

print('ntest_source', ntest_source)
average_time_per_sample_s = total_time_s / sample_count_s if sample_count_s > 0 else 0

print(f"Average prediction time per sample for source domain: {average_time_per_sample_s:.4f} seconds")
print(f"{args.model_type} done training for source domain; \nTest: {test_l2_s / ntest_source}, Test_nse: {test_nse_s / ntest_source}, Test_corr: {test_corr_s / ntest_source}, Test_csi_1: {test_csi_1_s / ntest_source}, Test_csi_2: {test_csi_2_s / ntest_source}, Test_csi_3: {test_csi_3_s / ntest_source}")

print('ntest_target', ntest_target)
average_time_per_sample_t = total_time_t / sample_count_t if sample_count_t > 0 else 0

print(f"Average prediction time per sample for Target domain: {average_time_per_sample_t:.4f} seconds")
print(f"{args.model_type} done training for Target domain; \nTest: {test_l2_t / ntest_target}, Test_nse: {test_nse_t / ntest_target}, Test_corr: {test_corr_t / ntest_target}, Test_csi_1: {test_csi_1_t / ntest_target}, Test_csi_2: {test_csi_2_t / ntest_target}, Test_csi_3: {test_csi_3_t / ntest_target}")
summary = f"Args: {str(args)}" \
          f"\nParameters: {complex_ct}" \
          f"\nTrain time: {train_time}" \
          f"\nMean epoch time: {train_times}" \
          f"\nMean inference time: {eval_times}" \
          f"\nNum inferences: {num_eval}" \
          f"\nTrain: {train_l2 / ntrain}" \
          f"\nValid: {valid_l2 / nvalid}" \
          f"\nTest for Source Domain: {test_l2_s / ntest_source}" \
          f"\nTest_nse for Source Domain: {test_nse_s/ntest_source}" \
          f"\nTest_corr for Source Domain: {test_corr_s/ntest_source}" \
          f"\nTest_csi_1 for Source Domain: {test_csi_1_s/ntest_source}" \
          f"\nTest_csi_2 for Source Domain: {test_csi_2_s/ntest_source}" \
          f"\nTest_csi_3 for Source Domain: {test_csi_3_s/ntest_source}" \
          f"\nTest for Target Domain: {test_l2_t / ntest_target}" \
          f"\nTest_nse for Target Domain: {test_nse_t/ntest_target}" \
          f"\nTest_corr for Target Domain: {test_corr_t/ntest_target}" \
          f"\nTest_csi_1 for Target Domain: {test_csi_1_t/ntest_target}" \
          f"\nTest_csi_2 for Target Domain: {test_csi_2_t/ntest_target}" \
          f"\nTest_csi_3 for Target Domain: {test_csi_3_t/ntest_target}"
if swe:
    summary += f"\nVorticity Test: {test_vort_l2 / ntest}" \
               f"\nPressure Test: {test_pres_l2 / ntest}"
txt = "results"
if args.txt_suffix:
    txt += f"_{args.txt_suffix}"
txt += ".txt"

with open(os.path.join(root, txt), 'w') as f:
    f.write(summary)
writer.flush()
writer.close()