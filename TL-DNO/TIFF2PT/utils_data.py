import torch
import matplotlib.pyplot as plt
import matplotlib.image as pm
import torch.nn as nn
import tifffile
from data import utils
import scipy.ndimage
from PIL import Image
import torch.nn.functional as F
from scipy import interpolate
from skimage.transform import resize
import imageio
import numpy as np
import io
import os


################################################################
# Dataset class
################################################################
dem_tif_path = 'Path/moa_bottom.tif'
man_path = 'Path/moa_rough.tif'
# Resize
def inter(array, size):
    h, w = array.shape
    new_h, new_w = np.floor_divide((h, w), size)
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    new_x = np.linspace(0, w - 1, new_w)
    new_y = np.linspace(0, h - 1, new_h)
    f = interpolate.interp2d(x, y, array, kind='linear')
    array_down = f(new_x, new_y)
    # array_down = resize(array, (new_h, new_w), order=1, anti_aliasing=True)
    return array_down

# DEM
dem_map = tifffile.imread(dem_tif_path)
print('dem_map', dem_map.shape)
def process_dem(dem_map):
    dem_map = np.nan_to_num(dem_map, nan=-99999)
    np_ma_map = np.ma.masked_array(dem_map, mask=(dem_map < -2000))
    mask_tensor = np_ma_map.mask
    np_ma_map = utils.fix_missing_values(np_ma_map)
    # print(np.min(np_ma_map))
    # print(np.max(np_ma_map))
    dem = torch.from_numpy(np_ma_map)
    return dem.float(), mask_tensor

def data_load(path, name, mask_tensor):
    t0 = 25
    dt0 = 300
    t00, tfinal = 0, (t0) * dt0
    dx = 30.0 * 16
    # # data
    h_gt = []
    qx_gt = []
    qy_gt = []
    for i in range(t00, tfinal, dt0):
        current_time = str(i)
        print('current_time', current_time)
        path_h = os.path.join(path, '%s_%s'%(name,current_time)+'H'+".tif")
        path_u = os.path.join(path, '%s_%s'%(name,current_time)+'U'+".tif")
        path_v = os.path.join(path, '%s_%s'%(name,current_time)+'V'+".tif")
        # path_h = path_h.replace("'", "\"")
        # print('path_h', path_h)
        h_current = Image.open(path_h)
        h_current = np.array(h_current)
        # h_current[mask_tensor] = np.nan
        # print('building_mask', np.sum(np.isnan(h_current)))
        # building_mask = ~mask_tensor & np.isnan(h_current)
        # building_mask = ~building_mask
        # h_current[~mask_tensor & np.isnan(h_current)] = 0.0
        # h_current = np.nan_to_num(h_current, nan=0.0)
        # h_current_wn = np.nan_to_num(h_current, nan=-99999)
        # h_current = np.ma.masked_array(h_current_wn, mask=(h_current_wn < -2000))
        # print(np.array_equal(mask_tensor, h_current.mask))
        h_current = torch.from_numpy(h_current)

        h_current = h_current.float()
        qx_current = Image.open(path_u)
        qx_current = np.array(qx_current)
        # qx_current = np.nan_to_num(qx_current, nan=0.0)
        # qx_current[~mask_tensor & np.isnan(qx_current)] = 0.0
        # qx_current_wn = np.nan_to_num(qx_current, nan=-99999)
        # qx_current = np.ma.masked_array(qx_current_wn, mask=(qx_current_wn < -2000))
        qx_current = torch.from_numpy(qx_current)
        qx_current = qx_current.float()
        qy_current = Image.open(path_v)
        qy_current = np.array(qy_current)
        # qy_current = np.nan_to_num(qy_current, nan=0.0)
        # qy_current[~mask_tensor & np.isnan(qy_current)] = 0.0
        # qy_current_wn = np.nan_to_num(qy_current, nan=-99999)
        # qy_current = np.ma.masked_array(qy_current_wn, mask=(qy_current_wn < -2000))
        qy_current = torch.from_numpy(qy_current)
        qy_current = qy_current.float()
        h_current = torch.unsqueeze(h_current, -1)
        qx_current = torch.unsqueeze(qx_current, -1)
        qy_current = torch.unsqueeze(qy_current, -1)
        h_gt.append(h_current)
        qx_gt.append(qx_current)
        qy_gt.append(qy_current)
    h_gt = torch.stack(h_gt, 0)
    u_gt = torch.stack(qx_gt, 0)
    v_gt = torch.stack(qy_gt, 0)
    print('len_supervised', h_gt.size())
    # pre
    pre = np.zeros((t0))
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".txt"):
                pre_path = os.path.join(root, file)
                print('pre_path',pre_path)
                with open(pre_path, 'r') as fil:
                    for line in fil:
                        condition, value = line.strip().split()
                        m = int(int(condition)/dt0)
                        if m <= t0:
                            pre[m] = value
    print('pre', pre)
    return h_gt, u_gt, v_gt, pre



class flood_data(torch.utils.data.Dataset):
    def __init__(self, path_root, T_in, T_out=None, train=True, strategy="markov", std=0.0):
        self.markov = strategy == "markov"
        self.teacher_forcing = strategy == "teacher_forcing"
        self.one_shot = strategy == "oneshot"
        # self.data = data[..., :(T_in + T_out)] if self.one_shot else data[..., :(T_in + T_out), :]
        self.data = self.load(path_root)
        self.nt = T_in + T_out
        self.T_in = T_in
        self.T_out = T_out
        self.num_hist = 1 if self.markov else self.T_in
        self.train = train
        self.noise_std = std

    def log_transform(self, data, eps=1e-2):
        return torch.log(1 + data/eps)

    def load(self, path_root):
        t0 = 25
        # days_train = 12
        # T = 86400
        m00 = dem_map.shape[0]
        n00 = dem_map.shape[1]
        h_gt_list = []
        u_gt_list = []
        v_gt_list = []
        z_list = []
        pre_list = []
        z_dem, mask_tensor = process_dem(dem_map)
        # z_dem = torch.nn.functional.normalize(z_dem)
        i = 0
        for root, directories, files in os.walk(path_root):
            for subdirectory in directories:
                path = os.path.join(root, subdirectory)
                name = os.path.basename(path)
                print(path)
                print('Name',name)
                h_gt, u_gt, v_gt, pre = data_load(path, name, mask_tensor)
                gridpre = torch.from_numpy(pre)
                # gridpre = self.log_transform(gridpre)
                gridpre = gridpre.reshape(t0, 1, 1, 1).repeat([1, m00, n00, 1])
                z = z_dem
                z = z.reshape(1, m00, n00, 1).repeat([t0, 1, 1, 1])
                print('DEM_shape', z.shape)
                data = torch.cat((h_gt, u_gt), dim=-1)
                data = torch.cat((data, v_gt), dim=-1)
                data = torch.cat((data, gridpre), dim=-1)
                data = torch.cat((data, z), dim=-1)
                data = data.permute(1, 2, 0, 3)
                print('data', data.shape)
                path_data = os.path.join(path_root, str(i) + ".pt")
                print(path_data)
                torch.save(data, path_data)
                i = i + 1
                # z = torch.unsqueeze(z, dim=0)
                # h_gt_list.append(h_gt)
                # u_gt_list.append(u_gt)
                # v_gt_list.append(v_gt)
                # pre_list.append(gridpre)
                # z_list.append(z)
        # data_h = torch.stack(h_gt_list, 0)
        # data_u = torch.stack(u_gt_list, 0)
        # data_v = torch.stack(v_gt_list, 0)
        # data_z = torch.stack(z_list, 0)
        # # data_h = torch.nn.functional.normalize(data_h)
        # data_r = torch.stack(pre_list, 0)
        # # data_r = torch.nn.functional.normalize(data_r)
        # z = process_dem(dem_map)
        # # z = torch.nn.functional.normalize(z)
        # # data_z = z.reshape(1, 1, m, n, 1).repeat([data_h.size(0), t0, m, n, 1])
        # data = torch.cat((data_h, data_u), dim=-1)
        # data = torch.cat((data, data_v), dim=-1)
        # data = torch.cat((data, data_r), dim=-1)
        # data = torch.cat((data, data_z), dim=-1)
        # print('data', data.shape)
        # data = data.permute(0, 2, 3, 1, 4)
        return data

    def __len__(self):
        if self.train:
            if self.markov:
                return len(self.data) * (self.nt - 1)
            if self.teacher_forcing:
                return len(self.data) * (self.nt - self.T_in)
        return len(self.data)

    def __getitem__(self, idx):
        if not self.train or not (self.markov or self.teacher_forcing): # full target: return all future steps
            pde = self.data[idx]
            if self.one_shot:
                x = pde[..., :self.T_in, :]
                x[..., :4] = self.log_transform(x[..., :4])
                x = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                y = pde[..., self.T_in:(self.T_in + self.T_out), :3]
            else:
                x = pde[..., (self.T_in - self.num_hist):self.T_in, :]
                x[..., :4] = self.log_transform(x[..., :4])
                y = pde[..., self.T_in:(self.T_in + self.T_out), :3]
            return x, y
        pde_idx = idx // (self.nt - self.num_hist) # Markov / teacher forcing: only return one future step
        t_idx = idx % (self.nt - self.num_hist) + self.num_hist
        pde = self.data[pde_idx]
        x = pde[..., (t_idx - self.num_hist):t_idx, :]
        x[..., :4] = self.log_transform(x[..., :4])
        y = pde[..., t_idx, :3]
        if self.noise_std > 0:
            x += torch.randn(*x.shape, device=x.device) * self.noise_std

        return x, y

################################################################
# Lploss: code from https://github.com/zongyi-li/fourier_neural_operator
################################################################
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        assert x.shape == y.shape and len(x.shape) == 3, "wrong shape"
        diff_norms = torch.norm(x - y, self.p, 1)
        y_norms = torch.norm(y, self.p, 1)

        if self.reduction:
            loss = (diff_norms/y_norms).mean(-1) # average over channel dimension
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

################################################################
# equivariance checks
################################################################
# function for checking equivariance to 90 rotations of a scalar field
def eq_check_rt(model, x, spatial_dims):
    model.eval()
    diffs = []
    with torch.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in range(len(spatial_dims)):
            for l in range(j + 1, len(spatial_dims)):
                dims = [spatial_dims[j], spatial_dims[l]]
                diffs.append([((out.rot90(k=k, dims=dims) - model(x.rot90(k=k, dims=dims))) / out.rot90(k=k, dims=dims)).abs().nanmean().item() * 100 for k in range(1, 4)])
    return torch.tensor(diffs).mean().item()

# function for checking equivariance to reflections of a scalar field
def eq_check_rf(model, x, spatial_dims):
    model.eval()
    diffs = []
    with torch.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in spatial_dims:
            diffs.append(((out.flip(dims=(j, )) - model(x.flip(dims=(j, )))) / out.flip(dims=(j, ))).abs().nanmean().item() * 100)
    return torch.tensor(diffs).mean().item()

################################################################
# grids
################################################################
class grid(torch.nn.Module):
    def __init__(self, twoD, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric", "None"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.include_grid = grid_type != "None"
        self.grid_dim = (1 + (not self.symmetric) + (not twoD)) * self.include_grid
        if self.include_grid:
            if twoD:
                self.get_grid = self.twoD_grid
            else:
                self.get_grid = self.threeD_grid
        else:
            self.get_grid = torch.nn.Identity()
    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)

    def threeD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z).reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy, gridz), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = torch.cat((gridx + gridy, gridz), dim=-1)
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)