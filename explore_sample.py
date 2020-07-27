import fastai.vision as fsi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Set the relevant paths
#path_sample = fsi.Path('/media/simoes/DATA/fastai/ISIC_skin_cancer_Data/TrainData/SmallSample')
path_sample = fsi.Path('/media/simoes/fastai/ISIC_skin_cancer_Data/TrainData/SmallSample')
#path_general = fsi.Path('/media/simoes/DATA/fastai/ISIC_skin_cancer_Data')
path_general = fsi.Path('/media/simoes/fastai/ISIC_skin_cancer_Data')

imgs = (path_sample).ls()
type(imgs)
print( '\nFiles in path: \n', imgs)

# Inspect the chunk of the ground truth csv corresponding to this data sample.
gt_df = pd.read_csv(path_general/'ISIC_2019_Training_GroundTruth.csv',  nrows=20, iterator=False)
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print('Ground truth dataframe for the first 20 pictures: \n', gt_df)

# Visualize a few images
_,axs = plt.subplots(2, 3, figsize=(9,9))
axs = axs.flatten()
for i,ax in zip(range(len(axs)),axs):
    img = fsi.open_image(imgs[i])
    img.show(ax=ax, title='fig {}'.format(i))

# Resizing first image - shorter side to 64px as a test; maintain ratio.
img = fsi.open_image(imgs[0])
def resize_specify_shorter(img: fsi.Image, new_shorter_side: int):
    'Resize image `img` so that the shorter side is resized to `new_shorter_side` and maintaining the original ratio.'
    longer_side = img.size[0] * (img.size[0]>=img.size[1]) + img.size[1] * (img.size[1]>img.size[0])
    shorter_side = img.size[0] * (img.size[0]<=img.size[1]) + img.size[1] * (img.size[1]<img.size[0])
    ratio = new_shorter_side/shorter_side
    new_img = fsi.deepcopy(img)
    new_img.resize(size=(3, new_shorter_side, int(longer_side*ratio)))
    return new_img

new_shorter_side = 64
_,axs = plt.subplots(2, 1, figsize=(9,9))
img.show(ax=axs[0], title='First image before scaling')
print('Size before scaling:', img.size)
img = resize_specify_shorter(img, new_shorter_side)
img.show(ax=axs[1], title='First image after scaling')
print('Size after scaling:', img.size)

# Randomly cropping the first image
tfm = fsi.transform.rand_crop()
fig,axs = plt.subplots(3, 3, figsize=(9,9))
fig.suptitle('fig 0, randomized crops')
axs = axs.flatten()
for ax in axs:
    cropped_img = img.apply_tfms(tfm, size=224)
    cropped_img.show(ax=ax) 

# Scale and randomly crop all images; 3 crops per image.
new_shorter_side = 224
n_crops = 3
open_imgs = [fsi.open_image(img_path) for img_path in imgs]
scaled_imgs = [resize_specify_shorter(img, new_shorter_side) for img in open_imgs]
crop_tfm = fsi.transform.rand_crop()
#Repeat each image `n_crops` times:
scaled_imgs = [x for tupl in zip(*[scaled_imgs for _ in range(n_crops)]) for x in tupl]
cropped_imgs = [img.apply_tfms(crop_tfm, size=new_shorter_side) for img in scaled_imgs]

fig,axs = plt.subplots(3, 3, figsize=(9,9))
fig.suptitle('fig 0,1 and 2 - scaled and cropped')
axs = axs.flatten()
for i,ax in zip(range(len(axs)), axs):
    cropped_img = cropped_imgs[i]
    cropped_img.show(ax=ax) 

#Save cropped images for later use.
cropped_img_path = fsi.Path('Imgs')
for idx,img in zip(range(len(cropped_imgs)), cropped_imgs):
    img.save(cropped_img_path/'cropped_img_{}.png'.format(idx))

###Missing: Colour correction and resizing test data as well.

# Data augmentation
#Will flip, rotate and change lighting. But no zooming.
tfms = fsi.get_transforms(do_flip=True, flip_vert=True, max_rotate=45., max_zoom=1., max_lighting=0.2, max_warp=0.1, p_affine=0.5, p_lighting=0.5)

#src = ( fsi.ImageList.from_folder(cropped_img_path)
#       .split_by_rand_pct(0.2)
#       .label_from_lists([(1, 0) for i in range(16*3)], [(1, 0) for i in range(4*3)]) )


gt_crop_df = gt_df.iloc[np.arange(len(gt_df) * 3) // 3]
gt_crop_df.index = range(len(gt_crop_df))
print('Ground truth df for cropped images:\n', gt_crop_df)

#src = ( fsi.ImageList.from_df(gt_df, path_sample, suffix='jpg')
#       .split_by_rand_pct(0.2)
#       .label_from_lists([(1, 0) for i in range(16*3)], [(1, 0) for i in range(4*3)]) )
#
#data = ( src.transform(tfms, size=224)
#        .databunch(bs=12).normalize(fsi.imagenet_stats) ) #Default batch size of 64 is too big for this small sample size.
#
#print('Number of training images in `data`: ', len(data.train_ds))
#print('Number of validation images in `data`: ', len(data.valid_ds))
#
##Visualize the data:
#data.show_batch(rows=3, figsize=(12,9))
#
       
###CURRENT


