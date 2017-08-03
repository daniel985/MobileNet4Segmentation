# MobileNet4Segmentation
MobileNet for foreground-background segmentation

1.You need to get tensorflow models:

git clone https://github.com/tensorflow/models ~/tensorflow_models

2.Download pretrained MobileNet model from here:

https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

3.Prepare 'RGBA' image dataset for train:

'A' is the mask for foreground-background

4.Use 'RGB' image for test

