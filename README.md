# Pixel2Mesh-Pytorch

This repository aims to implement the ICCV 2019 paper: [Mesh R-CNN](https://arxiv.org/pdf/1906.02739.pdf) in PyTorch.

## Requirements

- PyTorch 1.2
- \>= Python 3

## External Codes

- [pixel2mesh](https://github.com/nywang16/Pixel2Mesh): Base code of Vertice Align layer

## Datasets

We use the datasets as specified in the article

### ShapeNet

The dataset was obtained from [3D-R2N2](https://github.com/chrischoy/3D-R2N2).
Please respect the [ShapeNet license](https://shapenet.org/terms) while using.

### Pix3D

The dataset was obtained from[Pix3D](https://github.com/xingyuansun/pix3d).
Please respect the [Pix3D license](https://creativecommons.org/licenses/by/4.0/)

## Getting Started

### Data Preparation

\*\* as part of the data preparation for ShapeNet we render meshes
this process is slow and using a GPU is highly recommended

```
python download_dataset.py --download_path /path/to/save/datasets
```

this will download and prepare the dataset creating the following structure

<br>-/path/to/save/datasets/dataset<br>
----shapeNet<br>
-------ShapeNetRendering<br>
-------ShapeNetVox32<br>
-------ShapeNetMeshes<br>
-------shapenet.json<br>
----pix3d<br>
-------img<br>
-------mask<br>
-------model<br>
-------pix3d.json<br>

### Train

We provide 2 scripts in order to train the model
train_backbone in order to train a backbone feature extractor for the GCN
and train in order to train backbone+GCN at the same time

it is recommended to first train a backbone feature extractor before
training the full model

we also supprot multi-GPU training in the form of DataParallelisem,
if more than 1 GPU is detected.

to run for eg. a full model training run.

```
python train.py *training_args
```

The hyper-parameters can be changed from command. To get more help, please use

```
python train.py -h
```

### Validation

To evaluate the model on a dataset, please use the following command

```
python evaluate.py *evaluation_args
```

### Demo

to run inference on an image, please use the following command

```
python demo.py *demo_args
```

# Code Structure

----meshRCNN our main package containing architecture definitions<br>
-------layers.py implementation of all custom GCN layers<br>
-------loss_functions.py implementation of loss functions<br>
-------pix3d_model.py definition of Pix3D's backbone and GCN models<br>
-------shapenet_model.py definition of ShaPenet's backbone and GCN models<br>

-----dataParallel a package implementing custom scatter gather methods in order to run our model on multiple GPUS<br>

----data a module providing our custom datasets and data loading procedures<br>
----utils a package containing various methods to manipulate voxels and meshes,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;such as serialization,displaying and sampling
