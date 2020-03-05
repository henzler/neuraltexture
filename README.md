# Neural Texture

Official code repository for the paper:

**Learning a Neural 3D Texture Space from 2D Exemplars [CVPR, 2020]**

[Henzler](https://henzler.github.io), [J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/), [Ritschel](http://www.homepages.ucl.ac.uk/~ucactri/)

**[[Paper](https://geometry.cs.ucl.ac.uk/projects/2020/neuraltexture/paper_docs/neuraltexture.pdf)] [[Project page](https://geometry.cs.ucl.ac.uk/projects/2020/neuraltexture/)]**

## Data

We downloaded all our textures from [https://www.textures.com/](https://www.textures.com/). 
Due to licensing reasons we cannot provide the data for training, however, we provide pre-trained models under `trained_models` for the classes `wood, grass, marble, rust_paint`.

#### Inference
In order to evaluate textures, add the desired texture to the corresponding folder under `datasets/<class_name>/test` and use one of the pre-trained models under `trained_models/` and run the evaluation (see instructions below). We already provide some exemplars.

#### Training
For training you will need to provide data sets under `datasets/<your_folder>` and provide two subdirectories: `train` and `test`.
We provide `test` exemplars for `wood`, `grass`, `marble` and `rust_paint`. If you would like to train using these classes please add a `train` folder containing training data.
#####

## Prerequisites

 - Ubuntu 18.04
 - cuDNN 7
 - CUDA 10.1
 - python3+
 - pyTorch 1.4
 - Download pretrained models (optional)

### Install dependencies
 

```
cd code/
pip install -r requirements.txt

cd custom_ops/noise
# build cuda code for noise sampler
TORCH_CUDA_ARCH_LIST=<desired version> python setup.py install
```

### Download pre-trained models

```
sh download_pretrained_models.sh
```

#### Logs

To visualise pre-trained training logs run the following:
```
tensorboard --logdir=./trained_models
```

## Usage

### Config file

The config files are located in `code/configs/neural_texture`. In the following we give an explanation for the 
most important variables:
```
dim: 2 # choose between 2 and 3 for 2D and 3D.
dataset:
  path: '../datasets/wood' # set path 
  use_single: -1 # -1 = train entire data set | 0,1,2,... = for single training
```  

### Training
    
```
cd code/
python train_neural_texture.py --config_path=<path/to/config> --job_id=<your_id>
```

The default `config_path` is set to `configs/neural_texture/config_default.yaml`. The default `job_id` is set to `1`.

### Inference

```
cd code/
python test_neural_texture.py --trained_model_path=path/to/models
```

The default `trained_model_path` is set to `../trained_models`. The results are saved under `trained_model_path/{model}/results`


## Bibtex

If you use the code, please cite our paper:

```
@inproceedings{henzler2020neuraltexture,
    title={Learning a Neural 3D Texture Space from 2D Exemplars},
    author={Henzler, Philipp and Mitra, Niloy J and Ritschel, Tobias},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
    month={June},
    year={2020}
}
```


##### Side Note
Unlike reported in the paper the encoder network in this implementation uses a ResNet architecture as it stabilises training. 
