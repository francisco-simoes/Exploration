import fastai.vision as fsi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Set the relevant paths
path_sample = fsi.Path('/media/simoes/DATA/fastai/ISIC_skin_cancer_Data/TrainData/SmallSample')
path_general = fsi.Path('/media/simoes/DATA/fastai/ISIC_skin_cancer_Data')

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

# Randomly cropping the first image
tfm = fsi.transform.rand_crop()
img = fsi.open_image(imgs[0])
fig,axs = plt.subplots(3, 3, figsize=(9,9))
fig.suptitle('fig 0, randomized crops')
axs = axs.flatten()
for ax in axs:
    cropped_img = img.apply_tfms(tfm, size=224)
    cropped_img.show(ax=ax) 


###CURRENT



#Create transforms for data augmentation.
#(see https://docs.fast.ai/vision.transform.html)
transforms = fsi.get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
print(len(transforms))


#After untaring the training and set images into folders train-jpg and test-jpg, we can load them.
#Let's load the training images, label them using train_v2.csv and create a validation set with 20% of them.
np.random.seed(1)
src = (fsi.ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))
print('Number of training images in `src`: ', len(src.train))
print('Number of validation images in `src`: ', len(src.valid))

#Create ImageDataBunch object from src.
#These images will be randomly transformed using `transforms` every time we feed them to the NN.
#The `normalize` method normalizes the data using the means and std deviations of the RGB channels from the ImageNet dataset.
data = (src.transform(transforms, size=128)
        .databunch().normalize(fsi.imagenet_stats))
print('Number of validation images in `data`: ', len(data.train_ds))

#Visualize the data:
data.show_batch(rows=3, figsize=(12,9))

#Let's create a `Learner` object using the resnet50 architecture.
#The metrics will be `accuracy-thresh` and `fbeta` (F-score); both will accept the label activations above `thresh` as the prediction, and compare with the true labels.
architecture = fsi.models.resnet50
acc_thres = fsi.partial(fsi.accuracy_thresh, thresh=0.2) #partial() feeds the second argument to the function in the first argument, resulting in a new function.
f_score = fsi.partial(fsi.fbeta, thresh=0.2)

#Create the learner:
learner = fsi.cnn_learner(data, base_arch=architecture, metrics=[acc_thres, f_score])

#Use LR Finder to select a learning rate.
#It varies the learning rate while going through minibatches of the data during one epoch, and by ploting it we can select a learning rate.
learner.lr_find()
learner.recorder.plot()

#Select lr using the plot:
lr = 5e-02

#Train the model for five epochs, using a cyclical learning rate.
#(see https://iconof.com/1cycle-learning-rate-policy/)
learner.fit_one_cycle(cyc_len=5, max_lr=slice(lr))


learner.save('stage-1') #Saves state in a new models folder at the data folder.

learner.load('stage-1')
learner.unfreeze() #Model comes 'frozen' by default until the last group. Must unfreeze to fine tune.

learner.lr_find()
learner.recorder.plot()
#Select lr using the plot:
lr2 = 1e-05
learner.fit_one_cycle(cyc_len=1, max_lr=slice(lr2, lr/10))
learner.save('stage-2a')
learner.load('stage-2a')

learner.fit_one_cycle(cyc_len=4, max_lr=slice(lr2, lr/10))
learner.save('stage-2b')

# Save model; ready to be used.
learner.export('fitted.pkl')
fitted = fsi.load_learner(path, file='fitted.pkl')

# Visualize model in action on a few images.
n_images = 3 #number of train images to visualize and test.
imgs = [data.train_ds[i][0] for i in range(n_images)]
true_labels = [data.train_ds[i][1] for i in range(n_images)]
pred_labels = [fitted.predict(img, 0.3)[1] for img in imgs]
fig, ax = plt.subplots(n_images)
fig.suptitle('Some predictions in training images')
for k in range(n_images):
    ax[k].title.set_text('Prediction: ' + str(pred_labels[k]) + '\nTrue: ' + str(true_labels[k]))
    fsi.show_image(fitted.predict(imgs[k], 0.3)[0], ax=ax[k])

