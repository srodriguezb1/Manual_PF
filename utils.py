import os
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


def maskData(files, path, datatype=float):
  masks = [0]*(4*len(files))
  for i, datafile in enumerate(files):
    img= np.array(Image.open(path + '/' + datafile + ".png"))[:,:,2]
    masks[4*i] = np.array([#(img<82).astype(datatype),
                          (img==255).astype(datatype),
                          (img==82).astype(datatype)]).transpose([0,2,1])[:,0:512,300:1324]
    masks[4*i+1] = np.array([#(img<82).astype(datatype),
                          (img==255).astype(datatype),
                          (img==82).astype(datatype)]).transpose([0,2,1])[:,512:1024,300:1324]
    masks[4*i+2] = np.array([#(img<82).astype(datatype),
                          (img==255).astype(datatype),
                          (img==82).astype(datatype)]).transpose([0,2,1])[:,1024:1536,300:1324]
    masks[4*i+3] = np.array([#(img<82).astype(datatype),
                          (img==255).astype(datatype),
                          (img==82).astype(datatype)]).transpose([0,2,1])[:,1536:2048,300:1324]
  return masks

def ImageData(files, path):
  masks = [0]*(4*len(files))
  for i, datafile in enumerate(files):
    img= Image.open(path + '/' + datafile + ".jpg")
    masks[4*i] = np.array(img).transpose([1,0,2])[0:512,300:1324,:]/255
    masks[4*i+1] = np.array(img).transpose([1,0,2])[512:1024,300:1324,:]/255
    masks[4*i+2] = np.array(img).transpose([1,0,2])[1024:1536,300:1324,:]/255
    masks[4*i+3] = np.array(img).transpose([1,0,2])[1536:2048,300:1324,:]/255
  return masks

def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])

def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

def convert_mask(data):
  mask2 = np.zeros((data.shape[0], data.shape[2], data.shape[3],3))
  for i, image in enumerate(data):
    mat1 = (data[i,0,:,:]>0.9)
    mat2 = (data[i,1,:,:]>0.9)
    for k in range(mat1.shape[0]):
      for l in range(mat1.shape[1]):
        if mat1[k,l]:
          mask2[i,k,l] = np.array([255,255,255])
        else:
          mask2[i,k,l] = np.array([0, 0 ,0])
        if mat2[k,l]:
          mask2[i,k,l] = np.array([216, 67, 82])
  return mask2/255

def convert_result(mdata, p_conf):
  mask2 = np.zeros((mdata.shape[0], mdata.shape[2], mdata.shape[3],3))
  for i, image in enumerate(mdata):
    mat1 = (mdata[i,0,:,:])
    mat2 = (mdata[i,1,:,:])
    for k in range(mat1.shape[0]):
      for l in range(mat1.shape[1]):
        if (mat2[k,l]>p_conf and mat2[k,l]>mat1[k,l]):
          mask2[i,k,l] = np.array([216, 67, 82])
        if (mat1[k,l]>p_conf and mat1[k,l]>mat2[k,l]):
          mask2[i,k,l] = np.array([255,255,255])
        
  return mask2/255

def reverse_transform(inp):
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  inp = (inp * 255).astype(np.uint8)

  return inp