#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:58:06 2021

@author: liulu
"""

from torch.utils.data import DataLoader
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from models import get_model
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math, numpy, os
from dataio.loader.utils import write_nifti_img
from torch.nn import functional as F



def plotNNFilter(units, figure_id, interp='bilinear', colormap=cm.jet, colormap_lim=None):
    plt.ion()
    filters = units.shape[2]
    n_columns = round(math.sqrt(filters))
    n_rows = math.ceil(filters / n_columns) + 1
    fig = plt.figure(figure_id, figsize=(n_rows*3,n_columns*3))
    fig.clf()

    for i in range(filters):
        ax1 = plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(units[:,:,i].T, interpolation=interp, cmap=colormap)
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.colorbar()
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
   
        
def visualization(json_name):
    layer_name = 'attentionblock2'
    json_opts = json_file_to_pyobj(json_name)
    train_opts = json_opts.training
    
    # Setup the NN Model
    model = get_model(json_opts.model)
    save_directory = os.path.join(model.save_dir, train_opts.arch_type, layer_name); mkdirfun(save_directory)
    #epochs = range(485, 490, 3)
    att_maps = list()
    int_imgs = list()
    subject_id = int(1)
    

    #json_opts = json_opts._replace(model=json_opts.model._replace(which_epoch=epoch))
    model = get_model(json_opts.model)
    # Setup Dataset and Augmentation
    dataset_class = get_dataset(train_opts.arch_type)
    dataset_path = get_dataset_path(train_opts.arch_type, json_opts.data_path)
    dataset_transform = get_dataset_transformation(train_opts.arch_type, opts=json_opts.augmentation)
        
    # Setup Data Loader
    dataset = dataset_class(dataset_path, split='test', transform=dataset_transform['valid'])
    data_loader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=False)
        
    for iteration, (input_arr, input_meta, _) in enumerate(data_loader, 1):
        if iteration == subject_id:
        # load the input image into the model
            model.set_input(input_arr)
            inp_fmap, out_fmap = model.get_feature_maps(layer_name=layer_name, upscale=False)

            # Display the input image and Down_sample the input image
            orig_input_img = model.input.permute(2, 3, 4, 1, 0).cpu().numpy()
            upsampled_attention = F.upsample(out_fmap[1], size=input_arr.size()[2:], mode='trilinear').data.squeeze().permute(1,2,3,0).cpu().numpy()

            # Append it to the list
            int_imgs.append(orig_input_img[:,:,:,0,0])
            att_maps.append(upsampled_attention[:,:,:,1])

            # return the model
            model.destructor()
            
    # Write the attentions to a nifti image
    input_meta['name'][0] = str(subject_id) + '_img_2.nii.gz'
    int_imgs = numpy.array(int_imgs).transpose([1,2,3,0])
    write_nifti_img(int_imgs, input_meta, savedir=save_directory)

    input_meta['name'][0] = str(subject_id) + '_att_2.nii.gz'
    att_maps = numpy.array(att_maps).transpose([1,2,3,0])
    write_nifti_img(att_maps, input_meta, savedir=save_directory)        
        
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg visualization attention Function')

    parser.add_argument('-c', '--config', help='testing config file', required=True)
    args = parser.parse_args()

    visualization(args.config)
        
        
        
    

    
    
  


    
