import datetime
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils_data import flood_data, LpLoss, eq_check_rt, eq_check_rf



################################################################
# TIFF files to PT files
################################################################
full_data = None # for superres
Path = 'Path/to/TIFF'


# train_data = flood_data(path_root=Path_train, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
# ntrain = len(train_data)
# print('ntrain', ntrain)
valid_data = flood_data(path_root=Path_valid, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
nvalid = len(valid_data)
print('nvalid', nvalid)
test_data = flood_data(path_root=Path_test, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
ntest = len(test_data)
print('ntest', ntest)