# Iterative inference

## Summary of the idea

The idea would be to learn an energy function (or approximate gradient) to perform iterative inference at test time and refine the segmentation provided by a state-of-the-art Fully Convolutional Network (FCN). If we choose to address the problem with Denoising Autoencoders (DAE), we will learn a vector field that estimates the score of a data distribution. We could then use the learnt vector field to move the segmentation output of the FCN towards the low-dimensional manifold where the true ground truths lie.

Here is what we would do at training time:
1) Train a FCN to perform image segmentation.
2) Train a DAE that takes as input **x** (image) or **h** (representation) as well as **ỹ** (ground truth corrupted with Gaussian noise, or maybe output of fcn?) and reconstruct the clean version of **y** (ground truth).

Here is what we would do at test time:
1) Obtain FCN prediction **ŷ** for a given image **x**
```
ŷ = fcn(x)
```
2) Iterative inference:
```
for i in range(nb_iter):
     ŷ =   ŷ - ε dE(ŷ, x)/dŷ
```
where dE/dŷ is approximated by `-1/σ^2 (r(ŷ)- ŷ)`, where r is the reconstruction function of the DAE (that can take more arguments).

## Pretrained models

Pre-trained models for semantic segmentation and pre-trained DAE for iterative inference can be found here:
```
/data/lisatmp4/romerosa/itinf/
```

## Relevant papers
See wiki

## TODO code list
- [ x ] Check what is wrong with the context module
- [ x ] Add code to use DenseNet
- [ ] Add code to use FCN-FC-ResNet
- [ ] Check InverseLayer (ongoing, Adriana)

### TODO experiments list
- [ ] Try different h (so far, pool4 seems to be the best option)
- [ ] Try context module (ongoing, Adriana)
- [ ] Try different noise z (ongoing, Adriana)
- [ ] Explore more architectures
- [ ] Run CRF
- [ ] Experiments from ground truth instead of pre-trained network output (ongoing, Adriana)

## How to run experiments

**Using DenseNets**:
```
THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python train_dae.py
THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python iterative_inference.py
```

**For other networks**:
```
THEANO_FLAGS='device=cuda' python train_dae.py
THEANO_FLAGS='device=cuda' python iterative_inference.py
```
