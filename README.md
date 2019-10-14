# Pixel2Mesh-Pytorch

This repository aims to implement the ICCV 2019 paper: [Mesh R-CNN](https://arxiv.org/pdf/1906.02739.pdf) in PyTorch.

## Requirements

- PyTorch 1.2
- Torchvision 0.4.0
- Python 3.7
- Sklearn 0.21.3
- Matplotlib 3.1.0

we've provided an environment file in order to setup a conda environment
named mesh_rcnn

```
conda env create -f environment.yml
```

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

### Pretrained Models

We provide our pretrained models and their respective training statistics
via this [link:](https://drive.google.com/file/d/1tEB2weVl6wv2I0okIkEZtvSI1G7_16es/view?usp=sharing)<br>
`https://drive.google.com/file/d/1tEB2weVl6wv2I0okIkEZtvSI1G7_16es/view?usp=sharing`<br>

simply download and extract the file inside the project root,<br>
this will create a folder named checkpoints with the following files:<br>
----checkpoints <br>
-------shapenet.pth&emsp;&emsp; our shapenet model weights<br>
-------shapenet.st &emsp;&emsp;&emsp;our shapenet model training statistics<br>
-------pix3d.pth &emsp;&emsp;&emsp;our pix3d model weights<br>
-------pix3d.st &emsp;&emsp;&emsp;our pix3d model training statistics<br>

the shapenet model was trained on the airplane class<br>
and the pix3d model trained in the desk class

### Training

We provide 2 scripts in order to train the model,<br>
`train_backbone.py` in order to train a backbone feature extractor for the GCN,<br>
and `train.py` in order to train backbone+GCN at the same time

we also support multi-GPU training in the form of Data Parallelism,
if more than 1 GPU is detected.

to run for eg. a full model training run for pix3d:.

```
python train.py  -m Pix3D --threshold 0.2 --dataRoot *path to the directory with the json file* -b 4 --nEpoch 30
--optim SGD --lr 2e-3 --weightDecay 1e-4 --voxel 3.0 --residual  -c desk --train_backbone
--model_path *path to checkpoint.pth*
--train_ratio 0.7
```

#### Usage

```
usage: train.py [-h] --model {ShapeNet,Pix3D} [--featDim FEATDIM]
                [--model_path MODEL_PATH] [--backbone_path BACKBONE_PATH]
                [--num_refinement_stages NUM_REFINEMENT_STAGES]
                [--threshold THRESHOLD] [--voxel_only] [--residual]
                [--train_backbone] [--chamfer CHAMFER] [--voxel VOXEL]
                [--normal NORMAL] [--edge EDGE] [--backbone BACKBONE]
                [--num_sampels NUM_SAMPELS] [--train_ratio TRAIN_RATIO]
                [-c CLASSES] [--dataRoot DATAROOT] [--batchSize BATCHSIZE]
                [--workers WORKERS] [--nEpoch NEPOCH] [--optim {Adam,SGD}]
                [--weightDecay WEIGHTDECAY] [--lr LR]

GCN training script

optional arguments:
  -h, --help            show this help message and exit
  --model {ShapeNet,Pix3D}, -m {ShapeNet,Pix3D}
                        the model we wish to train
  --featDim FEATDIM     number of vertex features
  --model_path MODEL_PATH
                        path of a pretrained model to cintinue training
  --backbone_path BACKBONE_PATH, -bp BACKBONE_PATH
                        path of a pretrained backbone if we wish to continue
                        training from checkpoint must be provided with
                        GCN_path
  --num_refinement_stages NUM_REFINEMENT_STAGES, -nr NUM_REFINEMENT_STAGES
                        number of mesh refinement stages
  --threshold THRESHOLD, -th THRESHOLD
                        Cubify threshold
  --voxel_only          whether to return only the cubified mesh resulting
                        from cubify
  --residual            whether to use residual refinement for ShapeNet
  --train_backbone      whether to train the backbone in additon to the GCN
  --chamfer CHAMFER     weight of the chamfer loss
  --voxel VOXEL         weight of the voxel loss
  --normal NORMAL       weight of the normal loss
  --edge EDGE           weight of the edge loss
  --backbone BACKBONE   weight of the backbone loss
  --num_sampels NUM_SAMPELS
                        number of sampels to dataset
  --train_ratio TRAIN_RATIO
                        ration of samples used for training
  -c CLASSES, --classes CLASSES
                        classes of the exampels in the dataset
  --dataRoot DATAROOT   file root
  --batchSize BATCHSIZE, -b BATCHSIZE
                        batch size
  --workers WORKERS     number of data loading workers
  --nEpoch NEPOCH       number of epochs to train for
  --optim {Adam,SGD}    optimizer to use
  --weightDecay WEIGHTDECAY
                        weight decay for L2 loss
  --lr LR               learning rate
```

it is recommended to train a voxel only model before training the full GCN model<br>
by running `python train.py --voxel_only`<br>
because when the model is untrained it will predict huge graphs which can cause memory issues.

to run for eg. backbone only training:

```
python backbone_train.py --model ShapeNet --backbone_path *path to backbone.pth* -c airplane
--dataRoot *path to directory with json file* -b 32 --train_ratio 0.7 --nEpoch 100 --optim Adam
```

#### Usage

```
usage: train_backbone.py [-h] --model {ShapeNet,Pix3D}
                         [--backbone_path BACKBONE_PATH]
                         [--num_sampels NUM_SAMPELS]
                         [--train_ratio TRAIN_RATIO] [-c CLASSES]
                         [--dataRoot DATAROOT] [--batchSize BATCHSIZE]
                         [--workers WORKERS] [--nEpoch NEPOCH]
                         [--optim {Adam,SGD}] [--weightDecay WEIGHTDECAY]
                         [--lr LR]

backbone training script

optional arguments:
  -h, --help            show this help message and exit
  --model {ShapeNet,Pix3D}, -m {ShapeNet,Pix3D}
                        the backbone model we wish to train
  --backbone_path BACKBONE_PATH, -bp BACKBONE_PATH
                        path of a pretrained backbone if we wish to continue
                        training from checkpoint must be provided with
                        GCN_path
  --num_sampels NUM_SAMPELS
                        number of sampels to dataset
  --train_ratio TRAIN_RATIO
                        ration of samples used for training
  -c CLASSES, --classes CLASSES
                        classes of the exampels in the dataset
  --dataRoot DATAROOT   file root
  --batchSize BATCHSIZE, -b BATCHSIZE
                        batch size
  --workers WORKERS     number of data loading workers
  --nEpoch NEPOCH       number of epochs to train for
  --optim {Adam,SGD}    optimizer to use
  --weightDecay WEIGHTDECAY
                        weight decay for L2 loss
  --lr LR               learning rate
```

### Validation

To evaluate the model on a dataset, please use the following command

```
python eval_model.py e-m ShapeNet --model_path *path to checkpoint.pth* --residual --test_ratio 0.7 -c airplane
--dataRoot *path to directory with json file* -b 2 --output_path *path to save the output to*
```

#### Usage

```
usage: eval_model.py [-h] --model {ShapeNet,Pix3D} [--featDim FEATDIM]
                     [--model_path MODEL_PATH]
                     [--num_refinement_stages NUM_REFINEMENT_STAGES]
                     [--threshold THRESHOLD] [--residual]
                     [--test_ratio TEST_RATIO] [-c CLASSES]
                     [--dataRoot DATAROOT] [--batchSize BATCHSIZE]
                     [--workers WORKERS] [--output_path OUTPUT_PATH]

dataset evaluation script

optional arguments:
  -h, --help            show this help message and exit
  --model {ShapeNet,Pix3D}, -m {ShapeNet,Pix3D}
                        the model we wish to train
  --featDim FEATDIM     number of vertex features
  --model_path MODEL_PATH
                        the path to the model we wish to evaluate
  --num_refinement_stages NUM_REFINEMENT_STAGES, -nr NUM_REFINEMENT_STAGES
                        number of mesh refinement stages
  --threshold THRESHOLD, -th THRESHOLD
                        Cubify threshold
  --residual            whether to use residual refinement for ShapeNet
  --test_ratio TEST_RATIO
                        ratio of samples to test
  -c CLASSES, --classes CLASSES
                        classes of the exampels in the dataset
  --dataRoot DATAROOT   file root
  --batchSize BATCHSIZE, -b BATCHSIZE
                        batch size
  --workers WORKERS     number of data loading workers
  --output_path OUTPUT_PATH
                        path to output folder
```

### Demo

to run inference on an image, please use the following command

```
python demo.py demo.py -m ShapeNet --modelPath  --threshold 0.2
--imagePath *path to an image file to test on*
--savePath *path to save output files to*  --residual --show
```

it will create a .obj file containing the predicted meshes and .npy files containing the predicted voxel grids

#### Usage

```
usage: model inference script [-h] --model {ShapeNet,Pix3D}
                              [--featDim FEATDIM] --modelPath MODELPATH
                              [--num_refinement_stages NUM_REFINEMENT_STAGES]
                              [--threshold THRESHOLD] [--residual]
                              [--imagePath IMAGEPATH] [--savePath SAVEPATH]
                              [--show]

optional arguments:
  -h, --help            show this help message and exit
  --model {ShapeNet,Pix3D}, -m {ShapeNet,Pix3D}
                        the model to run the demo with
  --featDim FEATDIM     number of vertex features
  --modelPath MODELPATH
                        the path to find the trained model
  --num_refinement_stages NUM_REFINEMENT_STAGES, -nr NUM_REFINEMENT_STAGES
                        number of mesh refinement stages
  --threshold THRESHOLD, -th THRESHOLD
                        Cubify threshold
  --residual            whether to use residual refinement for ShapeNet
  --imagePath IMAGEPATH
                        the path to find the data
  --savePath SAVEPATH   the path to save the reconstructed meshes
  --show                whether to display the predicted voxels and meshes
```

### plot stats

to plot graphs of the training metrics from a .st file

```
python plot_stats.py -m Pix3D --statPath *path to stats.st file*
```

#### Usage

```
usage: plot_stats.py [-h] --model {ShapeNet,Pix3D} --statPath STATPATH

metrics plotting script

optional arguments:
  -h, --help            show this help message and exit
  --model {ShapeNet,Pix3D}, -m {ShapeNet,Pix3D}
                        the model we wish plot metrics for
  --statPath STATPATH   the path to the stats file(.st)
```

# Code Structure

----meshRCNN
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
our main package containing architecture definitions<br>
-------layers.py
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
implementation of all custom GCN layers<br>
-------loss_functions.py
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
implementation of loss functions<br>
-------pix3d_model.py
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
definition of Pix3D's backbone and GCN models<br>
-------shapenet_model.py
&nbsp;&nbsp;&nbsp;
definition of ShaPenet's backbone and GCN models<br>

-----dataParallel
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
a package implementing custom scatter gather methods,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
in order to run our model on multiple GPUS<br>

----data
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
a module providing our custom datasets and data loading procedures<br>

----utils
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
a package containing various methods to manipulate voxels and meshes,<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
such as serialization,displaying and sampling
