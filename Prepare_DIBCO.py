import numpy as np   
import utils as U   
   
Dataset_add = './DIBCO/'   
patch_h     = 128
patch_w     = 128
N_patches   = 100
    
(Images, Masks), (Te_d, Te_m ) = U.get_train_test(add = Dataset_add, Test_year = 2016)
patches_image, patches_masks   = U.extract_random(Images, Masks, patch_h, patch_w, N_patches)

np.save('patches_image', patches_image)
np.save('patches_masks', patches_masks)

print('Done')
