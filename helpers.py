import os
import numpy as np
import scipy
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float
import seaborn as sns


def my_label2rgb(labels, colors):
    """
    Converts labels to RGB

    Parameters
    ----------
    labels: labels of one image (0, 1)
    colors: colormap
    """
    output = np.zeros(labels.shape + (3,), dtype=np.float32)
    for i in range(len(colors)):
        output[(labels == i).nonzero()] = colors[i]
    return output


def my_label2rgboverlay(labels, colors, image, alpha=0.2):
    """
    Generates image with segmentation labels on top

    Parameters
    ----------
    labels:  labels of one image (0, 1)
    colors:  colormap
    image:   image (0, 1, c), where c=3 (rgb)
    alpha: transparency
    """
    image_float = gray2rgb(img_as_float(rgb2gray(image) if
                                        image.shape[2] == 3 else
                                        np.squeeze(image)))
    label_image = my_label2rgb(labels, colors)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


def save_img(image_batch, mask_batch, prediction_ii, prediction_fcn,
             out_images_folder, tag, void_label, colors):
    """
    Save image, segmentation, ground truth

    Parameters
    ----------
    image_batch: batch of images (b, c, 0, 1)
    mask_batch: batch of ground truth labels (b, 0, 1)
    prediction_fcn: batch of fcn predictions (before iter. inf.) (b, c, 0, 1) or (b, 0, 1)
    prediction_ii: batch of prediction after iterative inference (b, c, 0, 1) or (b, 0, 1)

    out_images_folder: folder where to save images
    tag: str, name of the batch
    void_label: list of void labels
    colors: 2d matrix of colors (nclasses, rgb)
    """

    # argmax predictions if needed
    if prediction_fcn.ndim == 4:
        prediction_fcn = prediction_fcn.argmax(1)
    if prediction_ii.ndim == 4:
        prediction_ii = prediction_ii.argmax(1)
    if mask_batch.ndim == 4:
        mask_batch = mask_batch.argmax(1)

    # apply void mask if needed
    if any(void_label):
        prediction_fcn[(mask_batch == void_label).nonzero()] = void_label[0]
        prediction_ii[(mask_batch == void_label)] = void_label[0]

    # fix img range if needed
    if image_batch.max() >= 1.0:
        image_batch /= 255

    color_map = [tuple(el) for el in colors]

    # prepare image to save for each element in batch
    images = []
    for j in xrange(prediction_ii.shape[0]):
        img = image_batch[j].transpose((1, 2, 0))

        # convert labels to rgb
        label_prediction_fcn = my_label2rgb(prediction_fcn[j], colors=color_map)
        label_prediction_ii = my_label2rgb(prediction_ii[j], colors=color_map)

        # put predictions on top of images
        pred_fcn_on_img = my_label2rgboverlay(prediction_fcn[j],
                                              colors=color_map,
                                              image=img, alpha=0.2)
        pred_ii_on_img = my_label2rgboverlay(prediction_ii[j],
                                              colors=color_map,
                                              image=img, alpha=0.2)
        # put gt on top of image
        mask_on_img = my_label2rgboverlay(mask_batch[j],
                                          colors=color_map,
                                          image=img, alpha=0.2)

        if img.shape[2] == 1:
            img = gray2rgb(img.squeeze())

        # combine images
        combined_image = np.concatenate((img, mask_on_img, pred_fcn_on_img,
                                         pred_ii_on_img), axis=1)

        # prepare filename and save image
        out_name = os.path.join(out_images_folder, tag + '_img' + str(j) +
                                '.png')
        scipy.misc.toimage(combined_image).save(out_name)
        images.append(combined_image)
    return images


def build_experiment_name(kind='fcn8', concat_h=[], optimizer='rmsprop',
                          training_loss=['crossentropy'],
                          learning_rate=0.0001, lr_anneal=0.99, data_aug=False,
                          weight_decay=0.0001, dropout=0.5, noise=0.0,
                          from_gt=False, temperature=1.0, n_filters=64,
                          conv_before_pool=1, skip=True, additional_pool=0,
                          unpool_type='standard'):
    """
    Build experiment name

    Parameters
    ----------
    dae_dict: dictionary
        Parameters of DAE
    training_loss: string
        Training loss
    data_aug: bool
        Whether or not we do data augmentation
    """

    all_concat_h = '_'.join(concat_h)
    all_loss = '_'.join(training_loss)

    exp_name = kind + '_' + all_concat_h

    if kind == 'standard':
        exp_name += '_f' + str(n_filters) + 'c' + \
            str(conv_before_pool) + 'p' + \
            str(additional_pool) + \
            ('_skip' if skip else '')
        exp_name += '_' + unpool_type

    exp_name += ('_dropout' + str(dropout) if dropout > 0. else '')
    exp_name += '_' +  all_loss
    exp_name += ('_fromgt' if from_gt else '_fromfcn8') + '_z' + \
                str(noise)
    exp_name += '_data_aug' if bool(data_aug) else ''
    exp_name += ('_T' + str(temperature)) if not from_gt else ''

    exp_name += ('_' + optimizer + '_lr' + str(learning_rate) + '_anneal' +
                str(lr_anneal) + '_decay' + str(weight_decay))

    print(exp_name)

    return exp_name


def print_results(st, rec, acc, jacc, nbatches):
        jacc_mean = np.mean(jacc[0, :] / jacc[1, :])
        print st
        print '    Loss: ' + str(rec/nbatches)
        print '    Acc: ' + str(acc/nbatches)
        print '    Jaccard: ' + str(jacc_mean)
