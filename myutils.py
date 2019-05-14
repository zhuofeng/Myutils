# -*- coding: utf8 -*-
import nibabel as nib
import os
import random
import math
from skimage.measure import block_reduce

import scipy
from scipy.ndimage.interpolation import zoom 
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
from PIL import Image
#import path
win_min=3000
win_max=12000

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, size, is_random=False):
    x = crop(x, wrg=64, hrg=64, is_random=is_random)
    return x

def crop_sub_imgs_fn3D(img, cropsize, is_random=False):
    imgshape = img.shape
    
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-cropsize)
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil((imgshape[2] - cropsize)/2)
    
    img = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    return img

def train_crop_sub_imgs_fn_andsmall3D(img, batchsize, cropsize, small_size, is_random=False):
    imgshape = img.shape
    imgbig = np.arange(batchsize*cropsize*cropsize*cropsize, dtype = 'float32').reshape(batchsize, cropsize, cropsize, cropsize, 1)
    imgsmall= np.arange(batchsize*small_size*small_size*small_size, dtype = 'float32').reshape(batchsize, small_size, small_size, small_size, 1)
    if is_random:
        for i in range(0, batchsize):
            x = random.randint(0, imgshape[0]-cropsize)
            y = random.randint(0, imgshape[1]-cropsize)
            z = random.randint(0, imgshape[2]-cropsize)
            imgbig[i,:,:,:,0] = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
            
    else:
        for i in range(0, batchsize):
            x = math.ceil((imgshape[0] - cropsize)/2)
            y = math.ceil((imgshape[1] - cropsize)/2)
            z = math.ceil((imgshape[2] - cropsize)/2)
            imgbig[i,:,:,:,0] = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    imgsmall = block_reduce(imgbig, block_size = (1,8,8,8,1), func=np.mean)
    imgsmall = zoom(imgsmall, (1,8.,8.,8.,1))
    
    return imgbig, imgsmall

def train_crop_both_imgs_fn_andsmall3D(imgbig, imgsmall, cropsize, is_random=False):
    imgshape = imgbig.shape
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-cropsize)
        imgpatchbig = imgbig[x:x+cropsize, y:y+cropsize, z:z+cropsize]
        imgpatchsmall = imgsmall[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil((imgshape[2] - cropsize)/2)
        imgpatchbig = imgbig[x:x+cropsize, y:y+cropsize, z:z+cropsize]
        imgpatchsmall = imgsmall[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    
    return imgpatchbig, imgpatchsmall

def train_crop_both_imgs_fn_andsmall(imgbig, imgsmall, cropsize, is_random=False):
    imgshape = imgbig.shape
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-cropsize)
        imgpatchbig = imgbig[x:x+cropsize, y:y+cropsize, z]
        imgpatchsmall = imgsmall[x:x+cropsize, y:y+cropsize, z]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil((imgshape[2] - cropsize)/2)
        imgpatchbig = imgbig[x:x+cropsize, y:y+cropsize, z]
        imgpatchsmall = imgsmall[x:x+cropsize, y:y+cropsize, z]
    
    return imgpatchbig, imgpatchsmall

def valid_crop_sub_imgs_fn_andsmall3D(img, xaxis, yaxis, zaxis, batchsize, cropsize, small_size, is_random=False):
    imgshape = img.shape #(1024, 1024, 64)
    imgbig = np.arange(batchsize*cropsize*cropsize*cropsize, dtype = 'float32').reshape(batchsize, cropsize, cropsize, cropsize, 1)
    imgsmall= np.arange(batchsize*small_size*small_size*small_size, dtype = 'float32').reshape(batchsize, small_size, small_size, small_size, 1)
    if is_random:
        for i in range(0, batchsize):
            x = random.randint(0, imgshape[0]-cropsize)
            y = random.randint(0, imgshape[1]-cropsize)
            z = random.randint(0, imgshape[2]-cropsize)
            imgbig[i,:,:,:,0] = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
            
    else:
        for i in range(0, batchsize):
            x = xaxis
            y = yaxis
            z = zaxis
            imgbig[i,:,:,:,0] = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    imgsmall = block_reduce(imgbig, block_size = (1,8,8,8,1), func=np.mean)
    imgsmall = zoom(imgsmall, (1,8.,8.,8.,1))
    
    return imgsmall

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    #print("before downsample:")
    #print(x.shape)
    #x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    #print(x.shape)
    #gaussian blurring
    #x = gaussian_filter(x, 2, order=0, output=None, mode='reflect')
    x = zoom(x, (0.125,0.125,1.0))  #8timesdownsampling
    #print(x.shape) #(96,96,3)
    return x

def downsample_zoom_fn(x):
    x = block_reduce(x, block_size = (8, 8, 1), func=np.mean)
    x = zoom(x, (8, 8, 1))
    return x
def downsample_fn2(x):
    x = zoom(x, (1,0.25,0.25))
    return x

def normalizationminmax1threhold(data):
    print('min/max data: {}/{} => {}/{}'.format(np.min(data),np.max(data),win_min,win_max))
    data = np.float32(data)
    data[data<win_min] = win_min
    data[data>win_max] = win_max
    data = data-np.min(data) 
    max = np.max(data)
    data = data - (max / 2.)
    data = data / max
    return data

def normalizationminmax1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = np.float32(data)
    max = np.max(data)
    min = np.min(data)
    data = data-min
    newmax = np.max(data)
    data = (data-(newmax/2)) / (newmax/2.)
    #print('this is the minmax of normalization')
    #print(np.max(data))
    #print(np.min(data))
    return data
def normalizationclinicalminmax1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = np.float32(data)
    max = np.max(data)
    min = np.min(data)
    data = data-min
    newmax = np.max(data)
    data = (data-(newmax/2)) / (newmax/2.)
    #print('this is the minmax of normalization')
    #print(np.max(data))
    #print(np.min(data))
    return data

def normalizationmicrominmax1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = np.float32(data)
    #data[data<0.] = 0.
    #data[data>15000.] = 15000.
    max = np.max(data)
    min = np.min(data)
    data = data-min
    newmax = np.max(data)
    data = (data-(newmax/2)) / (newmax/2.)
    #print('this is the minmax of normalization')
    #print(np.max(data))
    #print(np.min(data))
    return data

def normalizationmin0max1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = np.float32(data)
    data[data<0.] = 0.
    data[data>12000] = 12000
    #data[data<2000] = 2000
    #data = (data-(newmax/2)) / (newmax/2.)
    data = data / 12000.
    print('this is the minmax of normalization')
    print(np.max(data))
    print(np.min(data))
    return data

def normalizationtominmax(data):
    data[data<win_min] = win_min
    data[data>win_max] = win_max
    data = data-np.min(data)
    return data
def normalizationtoimg(data):
    print('min/max data: {}/{} => {}/{}'.format(np.min(data),np.max(data),win_min,win_max))
    data = data-np.min(data)
    data = data * (255.0/np.max(data))
    return data
def my_psnr(im1,im2):
    mse = ((im1 - im2) ** 2.).mean(axis=None)
    rmse = np.sqrt(mse)
    psnr = 20.*np.log10(1./rmse)
    return psnr

def my_ssim(im1,im2):
    mu1 = np.mean(im1)
    mu2 = np.mean(im2)
    c1 = 1e-4
    c2 = 1e-4
    sigma1 = np.std(im1)
    sigma2 = np.std(im2)
    
    im1 = im1 - mu1
    im2 = im2 - mu2
    cov12 = np.mean(np.multiply(im1,im2))
    
    
    ssim = (2*mu1*mu2+c1) * (2*cov12+c2) / (mu1**2+mu2**2+c1) / (sigma1**2 + sigma2**2 + c2)
    return ssim
def readnii(path):
    dpath = path
    img = nib.load(dpath)
    #print("this is the shape of img:{}".format(img.shape))
    #print(type(img)) #<class 'nibabel.nifti1.Nifti1Image'>
    #print("this is the shape of img.affine.shape:{}")
    #print("this is the header of img{}".format(img.header))
    data = img.get_fdata()
    #print(data.shape) #1024*1024*549
    #print(type(data))  #<class 'numpy.ndarray'>
    return data, img.header

def backtoitensity(path):
    
    #get the header
    correspondingimg = nib.load('/homes/tzheng/CTdata/CTMicroNUrespsurg/converted/DICOM_nulung026_cb_003_zf_ringRem.nii.gz')
    correspondingheader = correspondingimg.header
    empty_header = nib.Nifti1Header()
    empty_header = correspondingheader
    #print(empty_header)
    #print(correspondingimg.affine)
    #正规化导致neuves不能正常渲染
    
    thisimg = correspondingimg.get_fdata()
    valid_hr_slices = thisimg.shape[2]
    
    dpath = path
    img = nib.load(dpath)
    data = img.get_fdata()
    data = data * 12000.
    
    thisimg[160:810,160:810,int(valid_hr_slices*0.1/8)*8+10:int(valid_hr_slices*0.1/8)*8+410] = data[10:660,10:660,10:410]
    
    #saveimg = nib.Nifti1Image(data, correspondingimg.affine, empty_header)
    saveimg = nib.Nifti1Image(thisimg, correspondingimg.affine, empty_header)
    nib.save(saveimg, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest/SRbacktoitensity.nii.gz')
    
def mean_squared_error3d(output, target, is_mean=False, name="mean_squared_error"):
    if output.get_shape().ndims == 5:  # [batch_size, l, w, h, c]
        if is_mean:
            mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]), name=name)
        else:
            mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]), name=name)
    else:
        raise Exception("Unknow dimension")
    return mse

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))

def rotatenii_180():
    img = nib.load(config.VALID.Clinicalmedical_path)
    data = img.get_fdata()
    empty_header = nib.Nifti1Header()
    empty_header = img.header
    for i in range(0, data.shape[2]):
        tempimg = FZ(data[:,:,i])
        data[:,:,i] = tempimg
    data = nib.Nifti1Image(data, img.affine, empty_header) 
    nib.save(data, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest3D/rotatedclinical.nii.gz')

def crop_nifti_2D(img, cropsize, is_random=False):
    imgshape = img.shape
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-1)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil(imgshape[2] / 2)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    
    return imgpatch

def crop_nifti_withpos_2D(img, cropsize, is_random=False):
    imgshape = img.shape
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-1)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil(imgshape[2] / 2)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    
    return imgpatch, (x,y,z)

def dilatenifitimask():
    mask = nib.load('/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung036mask.nii.gz')
    maskdata = mask.get_fdata()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    
    for i in range(maskdata.shape[2]):
        maskdata[:,:,i] = cv2.dilate(maskdata[:,:,i], kernel)

    dilated = nib.Nifti1Image(maskdata, mask.affine, mask.header) 
    nib.save(dilated, '/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung036diatedmask.nii.gz')

#medium histogram matching
def Dual_subImage_HE(volume):
    print(volume.dtype) #float64
    newvolume = volume.copy()
    volume = volume.astype(np.int64) #转换成便于操作的方式
    #print(volume[200,200]) #2851
    #print(newvolume[200,200]) #2851.8970632851124
    median_value = np.median(volume)
    print('this is the median value')
    print(median_value)
    max_value = np.max(volume)
    min_value = np.min(volume)
    print(max_value) #不是整数，所以不能单独比较大小
    print(min_value)
    sum1 = np.sum(volume<median_value)
    sum2 = np.sum(volume>=median_value)
    print(sum1)
    print(sum2)
    #print(sum1+sum2) #=图像中所有的像素数
    #print(np.sum(volume<=2477)) #151184
    #os._exit(0)
    
    value1 = median_value - min_value
    value2 = max_value - min_value
    print(value1) #1408
    print(value2) #15797

    for i in range(int(min_value), int(median_value)):
        #print(i)
        #print(np.sum(volume<=i))
        #print(sum1)
        #print(np.sum(volume<=i) / sum1)
        #print(newvolume[volume==i])
        print((value1) * (np.sum(volume<=i) / sum1))
        newvolume[volume==i] = min_value + (value1) * (np.sum(volume<=i) / sum1) #在操作的时候，同一像素不能被操作两次。 所以用了newvolume
        #print(newvolume[volume==i])
    

    for j in range(int(median_value), int(max_value+1)):
        #print(j)
        #print(np.sum(np.logical_and(volume<=j, volume>=median_value)))
        #print(sum2)
        print((max_value - median_value) * (np.sum(np.logical_and(volume<=j, volume>=median_value)) / sum2))
        #print(type((max_value - median_value) * (np.sum(np.logical_and(volume<=j, volume>=median_value)) / sum2)))
        #print(newvolume.dtype)
        newvolume[volume==j] = median_value + (max_value - median_value) * (np.sum(np.logical_and(volume<=j, volume>=median_value)) / sum2)
        #print(newvolume[volume==j])
    #print(newvolume[volume==2494])
    print(np.max(newvolume)) #16867.0
    print(np.min(newvolume)) #1070.0093131548313
    print(np.median(newvolume)) #2495.6871844352218
    print(np.max(volume))
    print(np.min(volume))
    print(np.median(volume))
    print('this is the mean')
    print(np.mean(newvolume)) #5727.7661165424815
    print(np.mean(volume))
    return newvolume

def mean_subImage_HE(volume):
    print(volume.dtype) #float64
    newvolume = volume.copy()
    volume = volume.astype(np.int64) #转换成便于操作的方式
    #print(volume[200,200]) #2851
    #print(newvolume[200,200]) #2851.8970632851124
    median_value = np.mean(volume)
    print('this is the median value')
    print(median_value)
    max_value = np.max(volume)
    min_value = np.min(volume)
    print(max_value) #不是整数，所以不能单独比较大小
    print(min_value)
    sum1 = np.sum(volume<median_value)
    sum2 = np.sum(volume>=median_value)
    print(sum1)
    print(sum2)
    #print(sum1+sum2) #=图像中所有的像素数
    #print(np.sum(volume<=2477)) #151184
    #os._exit(0)
    
    value1 = median_value - min_value
    value2 = max_value - min_value
    print(value1) #1408
    print(value2) #15797

    for i in range(int(min_value), int(median_value)):
        #print(i)
        #print(np.sum(volume<=i))
        #print(sum1)
        #print(np.sum(volume<=i) / sum1)
        #print(newvolume[volume==i])
        print((value1) * (np.sum(volume<=i) / sum1))
        newvolume[volume==i] = min_value + (value1) * (np.sum(volume<=i) / sum1) #在操作的时候，同一像素不能被操作两次。 所以用了newvolume
        #print(newvolume[volume==i])
    

    for j in range(int(median_value), int(max_value+1)):
        #print(j)
        #print(np.sum(np.logical_and(volume<=j, volume>=median_value)))
        #print(sum2)
        print((max_value - median_value) * (np.sum(np.logical_and(volume<=j, volume>=median_value)) / sum2))
        #print(type((max_value - median_value) * (np.sum(np.logical_and(volume<=j, volume>=median_value)) / sum2)))
        #print(newvolume.dtype)
        newvolume[volume==j] = median_value + (max_value - median_value) * (np.sum(np.logical_and(volume<=j, volume>=median_value)) / sum2)
        #print(newvolume[volume==j])
    #print(newvolume[volume==2494])
    print(np.max(newvolume)) #16867.0
    print(np.min(newvolume)) #1070.0093131548313
    print(np.median(newvolume)) #2495.6871844352218
    print(np.max(volume))
    print(np.min(volume))
    print(np.median(volume))
    print('this is the mean')
    print(np.mean(newvolume)) #5727.7661165424815
    print(np.mean(volume))
    return newvolume

def median_filter2D(volume, ksize):
    d = int((ksize-1)/2)
    h,w = volume.shape[0], volume.shape[1]
    dst = volume.copy()
    for y in range(d, h - d):
        for x in range(d, w - d):
            dst[y][x] = np.median(volume[y-d:y+d+1, x-d:x+d+1])
    return dst

#histogram matching for lung CT image with a mask
def Dual_subImage_HE_MASK(volume, mask):
    print(volume.dtype) #float64
    newvolume = volume.copy()
    volume = volume.astype(np.int64) #转换成便于操作的方式
    mask = mask.astype(np.int64)
    median_value = np.median(volume[mask>0])
    print('this is the median value')
    print(median_value)
    max_value = np.max(volume[mask>0])
    min_value = np.min(volume[mask>0])
    print(max_value) #不是整数，所以不能单独比较大小
    print(min_value)
    sum1 = np.sum(volume[mask>0]<median_value)
    sum2 = np.sum(volume[mask>0]>=median_value)
    print(sum1)
    print(sum2)
    #print(sum1+sum2) #=图像中所有的像素数
    #print(np.sum(volume<=2477)) #151184
    
    value1 = median_value - min_value
    value2 = max_value - min_value
    print(value1) #1408
    print(value2) #15797

    for i in range(int(min_value), int(median_value)):
        #print(i)
        #print(np.sum(volume<=i))
        #print(sum1)
        #print(np.sum(volume<=i) / sum1)
        #print(newvolume[volume==i])
        print((value1) * (np.sum(np.logical_and(volume<=i, mask>0)) / sum1))
        newvolume[np.logical_and(volume==i, mask>0)] = min_value + (value1) * (np.sum(np.logical_and(volume<=i, mask>0)) / sum1)
        #print(newvolume[volume==i])


    for j in range(int(median_value), int(max_value+1)):
        #print(j)
        #print(np.sum(np.logical_and(volume<=j, volume>=median_value)))
        #print(sum2)
        print((max_value - median_value) * (np.sum(np.logical_and(np.logical_and(volume<=j, volume>=median_value), mask>0)) / sum2))
        #print(type((max_value - median_value) * (np.sum(np.logical_and(volume<=j, volume>=median_value)) / sum2)))
        #print(newvolume.dtype)
        newvolume[np.logical_and(volume==j, mask>0)] = median_value + (max_value - median_value) * (np.sum(np.logical_and(np.logical_and(volume<=j, volume>=median_value), mask>0)) / sum2)
        #print(newvolume[volume==j])
    #print(newvolume[volume==2494])
    print(np.max(newvolume)) #16867.0
    print(np.min(newvolume)) #1070.0093131548313
    print(np.median(newvolume)) #2495.6871844352218
    print(np.max(volume))
    print(np.min(volume))
    print(np.median(volume))
    print('this is the mean')
    print(np.mean(newvolume)) #5727.7661165424815
    print(np.mean(volume))
    return newvolume
    
#带mask的全局histogram matching
def all_HE_MASK(volume, mask):
    print(volume.dtype) #float64
    newvolume = volume.copy()
    volume = volume.astype(np.int64) #转换成便于操作的方式
    mask = mask.astype(np.int64)
    max_value = np.max(volume[mask>0])
    min_value = np.min(volume[mask>0])
    max_minus_min = max_value - min_value
    sum = np.sum(mask>0) #46102

    for i in range(int(min_value), int(max_value)):
        print(max_minus_min) #1877
        print(np.sum(np.logical_and(volume<=i, mask>0)))
        print(sum)
        print(np.sum(np.logical_and(volume<=i, mask>0)) / sum)
        print((max_minus_min) * (np.sum(np.logical_and(volume<=i, mask>0)) / sum))
        newvolume[np.logical_and(volume==i, mask>0)] = min_value + (max_value - min_value) * (np.sum(np.logical_and(volume<=i, mask>0)) / sum)

    #print(newvolume[volume==2494])
    print(np.max(newvolume)) #16867.0
    print(np.min(newvolume)) #1070.0093131548313
    print(np.median(newvolume)) #2495.6871844352218
    print(np.max(volume))
    print(np.min(volume))
    print(np.median(volume))
    return newvolume

#2个volume的分段灰度映射.将volume1映射到volume2.目前假设volume1是clinial CT, volume2是micro CT
def HE_2volumes(volume1, volume2, mask):
    newvolume1 = volume1 - np.min(volume1[mask>0])
    newvolume1[mask==0.] = -1000
    newvolume2 = volume2 - np.min(volume2)
    leftvalue1 = 0
    rightvalue1 = 0
    leftvalue2 = 0
    rightvalue2 = 0
    sum1 = np.sum(mask>0)
    sum2 = np.size(volume2)
    print('this is sum1 and sum2')
    print(sum1)
    print(sum2)
    #求某个范围内的值
    for i in range(0, int(np.max(newvolume1[mask>0]))):
        if np.sum(np.logical_and(newvolume1<=i, mask>0)) / sum1 > 0.05:
            leftvalue1 = i
            break
    for i in range(0, int(np.max(newvolume1[mask>0]))):
        if np.sum(np.logical_and(newvolume1<=i, mask>0)) / sum1 > 0.95:
            rightvalue1 = i
            break

    for i in range(0, int(np.max(newvolume2))):
        if np.sum(newvolume2<=i) / sum2 > 0.05:
            leftvalue2 = i
            break
    for i in range(0, int(np.max(newvolume2))):
        if np.sum(newvolume2<=i) / sum2 > 0.95:
            rightvalue2 = i
            break
    '''
    print('leftvalue1={}'.format(leftvalue1))
    print('rightvalue1={}'.format(rightvalue1))
    print('leftvalue2={}'.format(leftvalue2))
    print('rightvalue2={}'.format(rightvalue2))
    '''
    newvolume1[mask>0] = newvolume1[mask>0] * ((rightvalue2 - leftvalue2) / (rightvalue1 - leftvalue1))
    newvolume1[mask>0] = newvolume1[mask>0] + (np.mean(volume2) - np.mean(newvolume1[mask>0]))
    return newvolume1

if __name__ == '__main__':
    '''
    clinical_path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_slice.nii.gz'
    clinical = nib.load(clinical_path)
    clinical_data = clinical.get_fdata()
    clinical_mask_path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_mask_slice.nii.gz'
    clinical_mask = nib.load(clinical_mask_path)
    clinical_mask_data = clinical_mask.get_fdata()
    new_clinial_data = all_HE_MASK(clinical_data, clinical_mask_data)
    new_clinial_data = nib.Nifti1Image(new_clinial_data, clinical.affine, clinical.header)
    nib.save(new_clinial_data, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_all_HE_normaled.nii.gz')
    '''
    '''
    clinical_normalized_path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_all_HE_normaled.nii.gz'
    clinical_mask_path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_mask_slice.nii.gz'
    clinical_path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_slice.nii.gz'
    clinical = nib.load(clinical_path)
    clinical_data = clinical.get_fdata()
    clinical_mask_data = nib.load(clinical_mask_path).get_fdata()
    clinical_normalized_data = nib.load(clinical_normalized_path).get_fdata()

    clinical_normalized_data[clinical_mask_data<=0] = 0.
    clinical_data[clinical_mask_data<=0] = 0.
    clinical_normalized_data = nib.Nifti1Image(clinical_normalized_data, clinical.affine, clinical.header)
    clinical_data = nib.Nifti1Image(clinical_data, clinical.affine, clinical.header)
    nib.save(clinical_normalized_data, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_all_HE_normaled_lung.nii.gz')
    nib.save(clinical_data, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_lung.nii.gz')
    '''

    clinical_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung050.nii.gz'
    micro_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung050/nulung050_053_000_cropped_oneslice.nii.gz'
    clinical_mask_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung050closedmask.nii.gz'
    clinical = nib.load(clinical_path)
    clinical_data = clinical.get_fdata()
    micro_data = nib.load(micro_path).get_fdata()
    mask_data = nib.load(clinical_mask_path).get_fdata() 

    clinical_data = HE_2volumes(clinical_data, micro_data, mask_data)
    clinical =  nib.Nifti1Image(clinical_data, clinical.affine, clinical.header)
    nib.save(clinical, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CTdata/clinical_lung_HEmatched.nii.gz')
    