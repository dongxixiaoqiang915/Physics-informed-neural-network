import numpy as np
import random
import math
from collections import ChainMap
from sklearn.model_selection import *

def c2r(x, axis=1):
    
    dtype = np.float32 if axis == 1 else np.float64
    opt = 1
    
    if opt == 1:
        x = np.concatenate(x)
        x = np.array(x)
    
    else:
        n = x.ndim
        if axis < 0: axis = n + axis
        if axis < n:
            newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                       + tuple([i for i in range(axis, n-1)])
            x = x.transpose(newshape)
    shape = x.shape
            
    return x

def split_dataset(im,gt,ini,ind, split_part = 0.3, vt_part = 0.5):
    
    sample_index = random.randint(0,100)
    data = np.hstack((im,gt,ini,ind))
    train_set, test_set = train_test_split(data,test_size = split_part,random_state = None)
    valid_set, test_set = train_test_split(test_set,test_size = vt_part,random_state = None)
    
    return train_set, valid_set, test_set

def random_mini_batches(train_data, valid_data, test_data, mini_batch_size , seed = 0, shuffle = False):
    
    all_data = (train_data, valid_data, test_data) 
    l = [e for e in all_data]
    im,gt,ini,ind = [],[],[],[]

    for index,dataset in enumerate(l):
        im.append(dataset[0])
        gt.append(dataset[1])
        ini.append(dataset[2])
        ind.append(dataset[3])    

    im = c2r(im)
    gt = c2r(gt)
    ini = c2r(ini)
    ind = c2r(ind)
    train_set, valid_set, test_set = split_dataset(im,gt,ini,ind)
    
    m = train_set.shape[0]                  # number of training examples
    n = valid_set.shape[0]                  # number of testing examples
    channel_s = train_set.shape[1]
    channel_num = range(0,channel_s)
    train_mini_batches = []
    valid_mini_batches = []
    test_mini_batches = []
    np.random.seed(seed)
    
    if shuffle:
        permutation_m = list(np.random.permutation(m))
        permutation_n = list(np.random.permutation(n))
        train_set = train_set[permutation_m, ...].reshape((train_set.shape[0], channel_s, train_set.shape[2], train_set.shape[3]))
        valid_set = valid_set[permutation_n, ...].reshape((valid_set.shape[0], channel_s, valid_set.shape[2], valid_set.shape[3]))
        test_set = test_set[permutation_n, ...].reshape((test_set.shape[0], channel_s, test_set.shape[2], test_set.shape[3]))
    else:
        train_set = train_set
        valid_set = valid_set
        test_set = test_set

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = train_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[0]:channel_num[0]+1, ...]
        mini_batch_Y = train_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[1]:channel_num[1]+1, ...]
        mini_batch_Id = train_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[2]:channel_num[2]+1, ...]
        mini_batch_I0 = train_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[3]:channel_num[3]+1, ...]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_Id, mini_batch_I0)
        train_mini_batches.append(mini_batch)
        
    num_complete_minibatches = math.floor(n/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = valid_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[0]:channel_num[0]+1, ...]
        mini_batch_Y = valid_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[1]:channel_num[1]+1, ...]
        mini_batch_Id = valid_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[2]:channel_num[2]+1, ...]
        mini_batch_I0 = valid_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[3]:channel_num[3]+1, ...]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_Id, mini_batch_I0)
        valid_mini_batches.append(mini_batch)
        
    for k in range(0, num_complete_minibatches):
        mini_batch_X = test_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[0]:channel_num[0]+1, ...]
        mini_batch_Y = test_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[1]:channel_num[1]+1, ...]
        mini_batch_Id = test_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[2]:channel_num[2]+1, ...]
        mini_batch_I0 = test_set[k * mini_batch_size : k * mini_batch_size + mini_batch_size, channel_num[3]:channel_num[3]+1, ...]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_Id, mini_batch_I0)
        test_mini_batches.append(mini_batch)
        
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, ...]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, ...]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return train_mini_batches, valid_mini_batches, test_mini_batches

class DeepChainMap(ChainMap):
    'Variant of ChainMap that allows direct updates to inner scopes'

    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key):
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)

def crop_2D_image_force_fg(img, crop_size, valid_voxels):
    """
    img must be [c, x, y]
    img[-1] must be the segmentation with segmentation>0 being foreground
    :param img:
    :param crop_size:
    :param valid_voxels: voxels belonging to the selected class
    :return: desired image pairs
    """
    assert len(valid_voxels.shape) == 2

    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 1)
    else:
        assert len(crop_size) == (len(
            img.shape) - 1), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    # we need to find the center coords that we can crop to without exceeding the image border
    lb_x = crop_size[0] // 2
    ub_x = img.shape[1] - crop_size[0] // 2 - crop_size[0] % 2
    lb_y = crop_size[1] // 2
    ub_y = img.shape[2] - crop_size[1] // 2 - crop_size[1] % 2

    if len(valid_voxels) == 0:
        selected_center_voxel = (np.random.random_integers(lb_x, ub_x),
                                 np.random.random_integers(lb_y, ub_y))
    else:
        selected_center_voxel = valid_voxels[np.random.choice(valid_voxels.shape[1]), :]

    selected_center_voxel = np.array(selected_center_voxel)
    for i in range(2):
        selected_center_voxel[i] = max(crop_size[i] // 2, selected_center_voxel[i])
        selected_center_voxel[i] = min(img.shape[i + 1] - crop_size[i] // 2 - crop_size[i] % 2,
                                       selected_center_voxel[i])

    result = img[:, (selected_center_voxel[0] - crop_size[0] // 2):(
            selected_center_voxel[0] + crop_size[0] // 2 + crop_size[0] % 2),
             (selected_center_voxel[1] - crop_size[1] // 2):(
                     selected_center_voxel[1] + crop_size[1] // 2 + crop_size[1] % 2)]
    return result

