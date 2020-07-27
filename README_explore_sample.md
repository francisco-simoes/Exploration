# Data analysis test
We use the first 20 images of the 2019's ISIC training dataset to try out some data processing and augmentation based on the article by Zhou et al.

## Preprocessing the input images
We do the following:
- Resize the raw images such that the shorter side has 224px, and keeping the ratio.
- Randomly extract 224px by 224px crops.
- Data augmentation with flipping (horizontal and vertical), rotations and lighting adjustments.


