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

## TODO list
- [x] Code fcn8 model and script to train the model
- [x] Code DAE model and script to train the model
- [x] Code to perform iterative inference
- [x] Add entropy
- [x] Add possibility of having an intermediate layer as input (not only x)
- [x] Add possibility of having several intermediate layers as input
- [x] Add skip connections in DAE
- [x] Add WWAE pooling tracking
- [x] Code CRF
- [ ] Add possibility of having only y
- [ ] Add training DAE training from the output of FCN8 instead of GT.
- [ ] Add noise to skip connections
- [ ] Add BFGS optimization


### TODO experiments
**Camvid:**

| Layer h | Extra depth | Error | Input y | Skip | Upsampling |
|---------|-------------|-------|---------|------|------------|
| Pool 3  | +2          | CE    | FCN8    | No   | standard   |
| Pool 3  | +2          | MSE   | FCN8    | No   | standard   |
| Pool 3  | +2          | MSE   | GT      | No   | standard   |
| Pool 3  | +2          | MSE   | GT      | Yes  | standard   |
| Pool 3  | +2          | MSE   | FCN8    | Yes  | standard   |
| Pool 4  | +1          | MSE   | GT      | No   | standard   |
| Input   | +5          | MSE   | GT      | No   | standard   |
| Input   | +5          | MSE   | GT      | Yes  | standard   |
| Pool 1  | +4          | MSE   | GT      | No   | standard   |



**Future datasets:**
- [ ] PascalVOC
- [ ] Polyps 
- [ ] Nerve ultrasound?

## Some results
### CamVid
| **Ours** | Gl. Accuracy | Jaccard Ind. |
|-------------------|--------------|--------------|
| FCN-8 baseline     |88.06|57.03|
| FCN-8 + CRF     |||
| FCN-8 + DAE (input, 256)     |||
| FCN-8 + DAE (pool1, 512)     |||
| FCN-8 + DAE (pool3, 1024)     |88.48|57.78|
| FCN-8 + DAE (pool5, 4096)     |||
| FCN-8 + DAE (pool1, pool3, 1024)     |||
| FCN-8 + DAE (pool1, pool3, pool5, 4096)     |||


| **SOTA methods** | Gl. Accuracy | Jaccard Ind. |
|------------------|--------------|--------------|
|SegNet Basic          |82.8|46.3|
|SegNet                |88.6|50.2|
|Bayesian SegNet Basic |81.6|55.8|
|Reseg                 |**88.7**|58.8|
|Bayesian SegNet       |86.9|**63.1**|

### PascalVOC
| **Ours** | Gl. Accuracy | Jaccard Ind. |
|-------------------|--------------|--------------|
| FCN-8 baseline     |||


| **SOTA methods** | Gl. Accuracy | Jaccard Ind. |
|------------------|--------------|--------------|



