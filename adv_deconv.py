import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
# from PIL import Image
from functools import partial
import numpy as np
from pathlib import Path
import sys
import cv2
import os

from vgg import VGG
from vgg16_deconv import _vgg16Deconv
# from adversary import Attack

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
output_dir = Path('./output')
model_path = Path('epochs/vggcifar 100_3dpool_network_1.pth')
# parameter for ifgsm
eps = 0.03
iteration = 1


def fea_var_cuda(data):
    data_mean = data.mean(dim=0)
    temp = torch.zeros_like(data)
    for i in range(data.shape[0]):
        temp[i, :, :] = data_mean
    var = torch.sum((data - temp) ** 2, dim=0) / (data.shape[0] - 1)
    return var.sum() / var.numel()


def pic_var_cuda(data):
    data_mean = data.mean(dim=0)
    temp = torch.zeros_like(data)
    for i in range(data.shape[0]):
        temp[i, :, :, :] = data_mean
    var = torch.sum((data - temp) ** 2, dim=0) / (data.shape[0] - 1)
    return var.sum() / var.numel()


# Stroe all feature maps and max pooling locations during forward pass
def store_feat_maps(model):

    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool2d):
            model.feat_maps[key] = output[0]
            model.pool_locs[key] = output[1]
        else:
            model.feat_maps[key] = output

    for idx, layer in enumerate(model._modules.get('features')):
        layer.register_forward_hook(partial(hook, key=idx))


# Dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
testset = torchvision.datasets.CIFAR10(root='data/', train=False,
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

# Load the conv model
net = VGG('VGG16')
net = net.cuda()
criterion_no_constrain = nn.CrossEntropyLoss()
net.load_state_dict(torch.load(model_path))

net.eval()
store_feat_maps(net)    # store all feature maps and max pooling locations
#
# Build the deconv net
net_decocnv = _vgg16Deconv()
for idx, layer in enumerate(net.features):
    if isinstance(layer, nn.Conv2d):
        net_decocnv.features[net_decocnv.conv2deconv_indices[idx]].weight.data = layer.weight.data
        if idx in net_decocnv.conv2deconv_bias_indices:
            net_decocnv.features[net_decocnv.conv2deconv_bias_indices[idx]].bias.data = layer.bias.data
net_decocnv = net_decocnv.cuda()
net_decocnv.eval()

next_img = 1
# Test
for data in testloader:
    if next_img == 0:
        continue
    x, y_true = data
    x, y_ture = Variable(x), Variable(y_true)
    x = x.cuda()
    y_true = y_true.cuda()
    # print(x.shape)
    # output = net(x)

    # # Add attack
    # attack = Attack(net, criterion_no_constrain)
    # # r, loop_it, label_orig, label_pert, pert_x = attack.deepfool(x, net)
    # pert_x, h_adv, h = attack.i_fgsm(x, y_true, eps, iteration)
    # label_orig = h.max(1)[1]
    # label_pert = h_adv.max(1)[1]
    # str_label_orig = classes[label_orig]
    # str_label_pert = classes[label_pert]
    # print('[BEFORE]  label_orig : {:}'.format(str_label_orig))
    # print('[AFTER]   label_pert : {:}'.format(str_label_pert))
    # print('==============================')
    # save_image(x, output_dir.joinpath('ifgsm-legitimate-{:}(
    # e:{},i:{}).jpg'.format(str_label_orig, eps, iteration)))
    # save_image(pert_x, output_dir.joinpath('ifgsm-perturbed-{:}(
    # :{},i:{}).jpg'.format(str_label_pert, eps, iteration)))
    save_image(x, output_dir.joinpath('origin-y:{}.jpg'.format(
        y_true.cpu().item())))

    # Forward pass
    # origvspert = input('to choose view the clean img or the pert img \
    #     (0 for clean, 1 for pert, others to exit): ')
    # origvspert = int(origvspert)
    # if origvspert == 0:
    output, intermediate_out = net(x)
    # elif origvspert == 1:
    #     output, intermediate_out = net(pert_x)
    # else:
    #     sys.exit(0)
    print("conv-intermediate_out: %.4f" % intermediate_out.var().item())
    print('==================================================')
    # Deconv
    while True:
        # [1] chooes the certain feature map
        layer = input('which layer to view (0-44, -2 to exit): ')
        layer = int(layer)
        if layer == -1:
            next_img = 1
            break
        elif layer < 0:
            sys.exit(0)
        else:
            next_img = 0

        new_feat_map = net.feat_maps[layer].clone()   # (1, C, H, W)
        print("new_feat_map: ", new_feat_map.shape)
        map_channels = new_feat_map.shape[1]

        # [2] the top K activations in the layer-th feature map
        act_lst = []
        for i in range(0, map_channels):
            choose_map = new_feat_map[0, i, :, :]
            activation = torch.max(choose_map)
            act_lst.append(activation.item())

        # top 5
        act_lst = np.array(act_lst)
        activation_idxs = np.argsort(
            act_lst)[(map_channels - 5): map_channels]
        # print("activation_idxs: ", activation_idxs)

        var_feat_map = new_feat_map[:, :5, :, :]
        var_deconv_map = torch.zeros(5, 3, 32, 32)
        var_deconv_img = torch.zeros(5, 32, 32, 3).numpy()

        # for _, idx in enumerate(activation_idxs):
        indexx = [2, 22, 100, 110, 220]
        # print("indexx: ", indexx)
        # for i in range(5):
        for i, idx in enumerate(indexx):
            # idx = input("choose which map: ")
            # idx = int(idx)
            tmp_feat_map = new_feat_map.clone()
            choose_map = new_feat_map[0, idx, :, :]
            max_activation = torch.max(choose_map)

            if idx == 0:
                tmp_feat_map[:, 1:, :, :] = 0
            else:
                tmp_feat_map[:, :idx, :, :] = 0
                if idx != net.feat_maps[layer].shape[1] - 1:
                    tmp_feat_map[:, idx + 1:, :, :] = 0

            var_feat_map[:, i, :, :] = tmp_feat_map[:, idx, :, :]

            # [3] deconv
            # print('layer: ', layer)
            deconv_output = net_decocnv(tmp_feat_map, layer, net.pool_locs)
            var_deconv_map[i] = deconv_output

            img_deconv = deconv_output.data.cpu(
            ).numpy()[0].transpose(1, 2, 0)   # (H, W, C)
            for j in range(3):
                img_deconv[:, :, j] = (
                    img_deconv[:, :, j] - img_deconv[:, :, j].min()) / (
                        img_deconv[:, :, j].max() - img_deconv[:, :, j].min(
                        )) * 255
            img_deconv = img_deconv.astype(np.uint8)
            # print(img_deconv)
            var_deconv_img[i] = img_deconv

            deconv_path = os.path.join(output_dir, "y" + str(
                                       y_true.cpu().item()) + "-l" +
                                       str(layer) + "_f" + str(idx) + ".jpg")
            cv2.imwrite(deconv_path, img_deconv)

        var_map = fea_var_cuda(var_feat_map.squeeze())
        print("conv-var_map(5): %.4f" % var_map.item())
        var_img = pic_var_cuda(torch.from_numpy(var_deconv_img).squeeze())
        print("deconv-var_img(5): %.4f" % var_img.item())
        print('--------------------------------------------------')
        # var_pic = pic_var_cuda(var_deconv_map.squeeze())
        # print("deconv-var_pic(5): %.4f" % var_pic.item())
        # print('--------------------------------------------------')
