# Distillation of blazepose

# abstract 

This is BlazePose Distillation to student model for <b>being lighten with few dataset. And for getting almost equal or more high performance of teacher model.</b>

<b>This is overall stracture</b>
<img src="https://user-images.githubusercontent.com/48679574/191453093-08aec30a-259c-467e-a5d4-0a2b67377ac3.png" width="600px">





## FitNets-Based Distillation
This is kind of Distillation that use teacher model backbon embedding output for Distillation through Attention-Layer.

<img src="https://user-images.githubusercontent.com/48679574/190838105-0d255020-df53-4a81-9620-f5768c3cfa4a.png" width="400px">



# Distillation Performance (student model)

## teacher model 

<img src="https://user-images.githubusercontent.com/48679574/191453672-40cca430-ef08-4b47-827f-6a4e792e66f6.gif" width="250" height="250"/>



## student model

![pred](https://user-images.githubusercontent.com/48679574/191453738-c8aa27c6-0f35-4121-9d25-a479a87e8a13.gif)


## loss curve
<img src="https://user-images.githubusercontent.com/48679574/191453151-b311250e-bd3a-4ea9-ba9b-f5eed8a8000c.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/191453161-48a623e6-aa44-4dbb-ad93-503f57246561.png" width="400px">




# dataset
- [MPII Human Pose Models](https://pose.mpi-inf.mpg.de)
- [VGG Human Pose Estimation datasets](https://www.robots.ox.ac.uk/~vgg/data/pose/)

# References
- [tflite models to PyTorch](https://github.com/zmurez/MediaPipePyTorch)
- [Learning Efficient Object Detection Models with Knowledge Distillation](https://proceedings.neurips.cc/paper/2017/file/e1e32e235eee1f970470a3a6658dfdd5-Paper.pdf)
