# Image Segmentation by Iterative Inference from Conditional Score Estimation

## Abstract

Inspired by the combination of feedforward and iterative computations in the
visual cortex, and taking advantage of the ability of denoising autoencoders to
estimate the score of a joint distribution, we propose a novel approach to iterative
inference for capturing and exploiting the complex joint distribution of output
variables conditioned on some input variables. This approach is applied to image
pixel-wise segmentation, with the estimated conditional score used to perform
gradient ascent towards a mode of the estimated conditional distribution. This
extends previous work on score estimation by denoising autoencoders to the case
of a conditional distribution, with a novel use of a corrupted feedforward predictor
replacing Gaussian corruption.

[Link to paper](https://arxiv.org/abs/1705.07450)

## Experiments

### fcn8

```
parser.add_argument('-dataset',
                    type=str,
                    default='camvid',
                    help='Dataset.')
parser.add_argument('-segmentation_net',
                    type=str,
                    default='fcn8',
                    help='Segmentation network.')
parser.add_argument('-train_dict',
                    type=dict,
                    default={'learning_rate': 0.001, 'lr_anneal': 0.99,
                             'weight_decay': 0.0001, 'num_epochs': 500,
                             'max_patience': 100, 'optimizer': 'rmsprop',
                             'batch_size': [10, 10, 10],
                             'training_loss': ['crossentropy',
                                               'squared_error'],
                             'lmb': 1, 'full_im_ft': False},
                    help='Training configuration')
parser.add_argument('-dae_dict',
                    type=dict,
                    default={'kind': 'standard', 'dropout': 0, 'skip': True,
                             'unpool_type': 'trackind', 'noise': 0.5,
                             'concat_h': ['pool4'], 'from_gt': False,
                             'n_filters': 64, 'conv_before_pool': 1,
                             'additional_pool': 2, 'temperature': 1.0,
                             'path_weights': '',  'layer': 'probs_dimshuffle',
                             'exp_name': 'final_', 'bn': 0},
                    help='DAE kind and parameters')
parser.add_argument('-data_augmentation',
                    type=dict,
                    default={'crop_size': (224, 224),
                             'horizontal_flip': 0.5,
                             'fill_mode':'constant'
                            },
                    help='Dictionary of data augmentation to be used')
parser.add_argument('-train_from_0_255',
                    type=bool,
                    default=False,
                    help='Whether to train from images within 0-255 range')

```

### DenseNets

Arguments to use during training/iterative inference:

```
parser.add_argument('-dataset',
                    type=str,
                    default='camvid',
                    help='Dataset.')
parser.add_argument('-segmentation_net',
                    type=str,
                    default='densenet',
                    help='Segmentation network.')
parser.add_argument('-train_dict',
                    type=dict,
                    default={'learning_rate': 0.001, 'lr_anneal': 0.99,
                             'weight_decay': 0.0001, 'num_epochs': 500,
                             'max_patience': 100, 'optimizer': 'rmsprop',
                             'batch_size': [10, 10, 10],
                             'training_loss': ['crossentropy',
                                               'squared_error'],
                             'lmb': 1, 'full_im_ft': False},
                    help='Training configuration')
parser.add_argument('-dae_dict',
                    type=dict,
                    default={'kind': 'standard', 'dropout': 0, 'skip': True,
                             'unpool_type': 'trackind', 'noise': 0.5,
                             'concat_h': ['pool4'], 'from_gt': False,
                             'n_filters': 64, 'conv_before_pool': 1,
                             'additional_pool': 2, 'temperature': 1.0,
                             'path_weights': '',  'layer': 'probs_dimshuffle',
                             'exp_name': 'final_', 'bn': 0},
                    help='DAE kind and parameters')
parser.add_argument('-data_augmentation',
                    type=dict,
                    default={'crop_size': (224, 224),
                             'horizontal_flip': 0.5,
                             'fill_mode':'constant'
                            },
                    help='Dictionary of data augmentation to be used')
parser.add_argument('-train_from_0_255',
                    type=bool,
                    default=False,
                    help='Whether to train from images within 0-255 range')

```

## How to run experiments

**For DenseNets**:
```
THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python train_dae.py
THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python iterative_inference_valid.py
THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python iterative_inference.py
```

**For other networks**:
```
THEANO_FLAGS='device=cuda' python train_dae.py
THEANO_FLAGS='device=cuda' python iterative_inference_valid.py
THEANO_FLAGS='device=cuda' python iterative_inference.py
```

**Summary**:

1) train_dae.py will train the denoising auto-encoder
2) iterative_inference_valid.py will cross-validate the number of iterations and step to use at inference time
3) with ```step``` and ```num_iter``` found in 2., iterative_inference.py will report the final results on the test set

We used theano commit ddafc3e2c457a36871263b5549f916f821a67c29 and lasagne commit 45bb5689f0b2edb7114608e88305e8074d29bbe7.
