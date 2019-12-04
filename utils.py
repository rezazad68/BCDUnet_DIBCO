import numpy as np
import glob
import os 
from  PIL import Image
import random

## Implemented by Reza Azad
## Function to read the dataset
def Get_files(Dataset_add, year, DIBCO):
    Images = []
    Masks  = []
    
    if year == 2009:
       for idy in range (2):
           DIBCO_filed = DIBCO['2009_d'+str(idy+1)]
           DIBCO_files = DIBCO['2009_s'+str(idy+1)]
           add1 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/' + '*.bmp')
           add2 = (Dataset_add + str(year)+'/'+DIBCO_files[0] +'/')
           image_paths = glob.glob(add1)
           for idx in range (len(image_paths)):
               seg_add   = image_paths[idx]
               seg_add   = (add2 + seg_add[len(seg_add)-7: len(seg_add)-4] + '.tiff')
               img = np.asarray(Image.open(image_paths[idx]).convert('L'))
               seg = ((np.asarray(Image.open(seg_add).convert('L'))) > 0.)*255
               Images.append(img)
               Masks.append(seg)
       
    elif year == 2010:
       DIBCO_filed = DIBCO['2010_d']
       DIBCO_files = DIBCO['2010_s']
       add1 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/' + '*.*')
       add2 = (Dataset_add + str(year)+'/'+DIBCO_files[0] +'/')
       image_paths = glob.glob(add1)
       for idx in range (len(image_paths)):
           seg_add   = image_paths[idx]
           seg_add   = (add2 + seg_add[len(seg_add)-7: len(seg_add)-4] + '_estGT.tiff')
           img = np.asarray(Image.open(image_paths[idx]).convert('L'))
           seg = ((np.asarray(Image.open(seg_add).convert('L'))) > 0.)*255
           Images.append(img)
           Masks.append(seg)
                      
    elif year == 2011:
       suf = ['h','p']
       for idy in range (2):
           DIBCO_filed = DIBCO['2011_'+suf[idy]]
           add1 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/' + '*.png')
           add2 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/')
           image_paths = glob.glob(add1)           
           for idx in range (len(image_paths)):
               seg_add   = image_paths[idx]
               seg_add   = (add2 + seg_add[len(seg_add)-7: len(seg_add)-4] + '_GT.tiff')
               img = np.asarray(Image.open(image_paths[idx]).convert('L'))
               seg = ((np.asarray(Image.open(seg_add).convert('L'))) > 0.)*255
               Images.append(img)
               Masks.append(seg)
               
    elif year == 2012:
       DIBCO_filed = DIBCO['2012_d']
       add1 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/' + '*.png')
       add2 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/')
       image_paths = glob.glob(add1)
       for idx in range (len(image_paths)):
           seg_add   = image_paths[idx]
           seg_add   = (add2 + seg_add[len(seg_add)-7: len(seg_add)-4] + '_GT.tif')
           img = np.asarray(Image.open(image_paths[idx]).convert('L'))
           seg = ((np.asarray(Image.open(seg_add).convert('L'))) > 0.)*255
           Images.append(img)
           Masks.append(seg)
           
    elif year == 2013:
       DIBCO_filed = DIBCO['2013_d']
       DIBCO_files = DIBCO['2013_s']
       add1 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/' + '*.*')
       add2 = (Dataset_add + str(year)+'/'+DIBCO_files[0])
       image_paths = glob.glob(add1)
       for idx in range (len(image_paths)):
           seg_add   = image_paths[idx]
           st = 1
           en = 1
           while seg_add[len(seg_add)- en] != '/':
                 en += 1
           while seg_add[len(seg_add)- st] != '.':
                 st += 1
           seg_add = seg_add[len(seg_add)-en: len(seg_add)-st]      
           seg_add   = (add2+seg_add+ '_estGT.tiff')
           
           img = np.asarray(Image.open(image_paths[idx]).convert('L'))
           seg = ((np.asarray(Image.open(seg_add).convert('L'))) > 0.)*255
           Images.append(img)
           Masks.append(seg)                          
                      
    elif year == 2014:
       DIBCO_filed = DIBCO['2014_d']
       DIBCO_files = DIBCO['2014_s']
       add1 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/' + '*.png')
       add2 = (Dataset_add + str(year)+'/'+DIBCO_files[0] +'/')
       image_paths = glob.glob(add1)
       for idx in range (len(image_paths)):
           seg_add   = image_paths[idx]
           seg_add   = (add2 + seg_add[len(seg_add)-7: len(seg_add)-4] + '_estGT.tiff')
           img = np.asarray(Image.open(image_paths[idx]).convert('L'))
           seg = ((np.asarray(Image.open(seg_add).convert('L'))) > 0.)*255
           Images.append(img)
           Masks.append(seg)
           
    elif year == 2016:
       DIBCO_filed = DIBCO['2016_d']
       DIBCO_files = DIBCO['2016_s']
       add1 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/' + '*.bmp')
       add2 = (Dataset_add + str(year)+'/'+DIBCO_files[0])
       image_paths = glob.glob(add1)
       for idx in range (len(image_paths)):
           seg_add   = image_paths[idx]
           st = 1
           en = 1
           while seg_add[len(seg_add)- en] != '/':
                 en += 1
           while seg_add[len(seg_add)- st] != '.':
                 st += 1
           seg_add = seg_add[len(seg_add)-en: len(seg_add)-st]      
           seg_add   = (add2+seg_add+ '_gt.bmp')
           
           img = np.asarray(Image.open(image_paths[idx]).convert('L'))
           seg = ((np.asarray(Image.open(seg_add).convert('L'))) > 0.)*255
           Images.append(img)
           Masks.append(seg)  
           
    elif year == 2017:
       DIBCO_filed = DIBCO['2017_d']
       DIBCO_files = DIBCO['2017_s']
       add1 = (Dataset_add + str(year)+'/'+DIBCO_filed[0] +'/' + '*.bmp')
       add2 = (Dataset_add + str(year)+'/'+DIBCO_files[0])
       image_paths = glob.glob(add1)
       for idx in range (len(image_paths)):
           seg_add   = image_paths[idx]
           st = 1
           en = 1
           while seg_add[len(seg_add)- en] != '/':
                 en += 1
           while seg_add[len(seg_add)- st] != '.':
                 st += 1
           seg_add = seg_add[len(seg_add)-en: len(seg_add)-st]      
           seg_add   = (add2+seg_add+ '_gt.bmp')
           
           img = np.asarray(Image.open(image_paths[idx]).convert('L'))
           seg = ((np.asarray(Image.open(seg_add).convert('L'))) > 0.)*255
           Images.append(img)
           Masks.append(seg)                                             

    else:
       
       print('Dataset for this year is not provided')   


    return  Images, Masks


## DIBCO dataset
def get_DIBCO_info():
    DIBCO = {}
    DIBCO['2009_d1'] = ['DIBC02009_Test_images-handwritten']
    DIBCO['2009_s1'] = ['DIBCO2009-GT-Test-images_handwritten']
    DIBCO['2009_d2'] = ['DIBCO2009_Test_images-printed']
    DIBCO['2009_s2'] = ['DIBCO2009-GT-Test-images_printed']

    DIBCO['2010_d'] = ['DIBC02010_Test_images']
    DIBCO['2010_s'] = ['DIBC02010_Test_GT']

    DIBCO['2011_h'] = ['DIBCO11-handwritten']
    DIBCO['2011_p'] = ['DIBCO11-machine_printed']

    DIBCO['2012_d'] = ['H-DIBCO2012-dataset']

    DIBCO['2013_d'] = ['OriginalImages']
    DIBCO['2013_s'] = ['GTimages']

    DIBCO['2014_d'] = ['original_images']
    DIBCO['2014_s'] = ['gt']

    DIBCO['2016_d'] = ['DIPCO2016_dataset']
    DIBCO['2016_s'] = ['DIPCO2016_Dataset_GT']

    DIBCO['2017_d'] = ['Dataset']
    DIBCO['2017_s'] = ['GT']
    
    return DIBCO


def get_train_test(add, Test_year = 2009):
    DIBCO = get_DIBCO_info()
    Tr_years = [2009, 2010, 2011 , 2012, 2013 , 2014, 2016 , 2017]
    Tr_years.remove(Test_year)
    Tr_years = np.array(Tr_years)
    Images, Masks = [], []
    for idx in range (Tr_years.shape[0]):
        T1, T2 = [], []
        T1, T2 = Get_files(add, Tr_years[idx], DIBCO)
        for idy in range(len(T1)):
            Images.append(T1[idy])
            Masks .append(T2[idy])
    Te_d, Te_m = Get_files(add, Test_year, DIBCO)
        
    return (Images, Masks), (Te_d, Te_m )



def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches):

    patches_image = np.empty((N_patches * len(full_imgs), patch_h, patch_w))
    patches_masks = np.empty((N_patches * len(full_imgs), patch_h, patch_w))
    T = 0
    for idx in range(len(full_imgs)):
        IMG = full_imgs[idx]
        MSK = full_masks[idx]
        for idy in range (N_patches):
            x_rand = random.randint(0, IMG.shape[0]- (patch_h+1))
            y_rand = random.randint(0, IMG.shape[1]- (patch_w+1))
            
            patches_image[T] = IMG[x_rand: x_rand + patch_h, y_rand: y_rand + patch_w]
            patches_masks[T] = MSK[x_rand: x_rand + patch_h, y_rand: y_rand + patch_w]
            T += 1

    return patches_image, patches_masks

def get_test_data(add, Test_year):
    DIBCO = get_DIBCO_info()
    Te_d, Te_m = Get_files(add, Test_year, DIBCO)
    return Te_d, Te_m
    
#Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    img_h = full_imgs.shape[0]  #height of the full image
    img_w = full_imgs.shape[1] #width of the full image
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    idx = 0
    patches = np.empty((N_patches_img, patch_h, patch_w))
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            patches[idx]= full_imgs[h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
            idx +=1
    assert (idx == N_patches_img)
    new_h = (h*stride_h)+patch_h
    new_w = (w*stride_w)+patch_w
    
    
    return patches , new_h, new_w


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
 
    full_prob = np.zeros((img_h, img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum  = np.zeros((img_h, img_w))
    idx  = 0
    
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            full_prob[h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w] += preds[idx]
            full_sum [h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w] += 1
            idx +=1
    assert(idx == preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0

    return final_avg    


       