import os
import numpy as np
import scipy
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float
import seaborn as sns


# TODO: test with new code
def my_label2rgb(labels, colors):
    """
    Converts labels to RGB

    Parameters
    ----------
    labels:
    colors:
    """
    output = np.zeros(labels.shape + (3,), dtype=np.float32)
    for i in range(len(colors)):
        output[(labels == i).nonzero()] = colors[i]
    return output


# TODO: test with new code
def my_label2rgboverlay(labels, colors, image, alpha=0.2):
    """
    Generates image with segmentation labels on top

    Parameters
    ----------
    labels:
    colors:
    image:
    alpha: transparency
    """
    image_float = gray2rgb(img_as_float(rgb2gray(image) if
                                        image.shape[2] == 3 else
                                        np.squeeze(image)))
    label_image = my_label2rgb(labels, colors)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


# TODO: test with new code
def save_img(image_batch, mask_batch, output, output_old, out_images_folder,
             n_classes, tag, void_label, colors):
    """
    Save image, segmentation, ground truth

    Parameters
    ----------
    image_batc:
    mask_batch:
    output:
    output_old:
    out_images_folder:
    n_classes:
    tag:
    void_label:
    colors:
    """

    if output.ndim == 4:
        output = output.argmax(1)

    if output_old.ndim == 4:
        output_old = output_old.argmax(1)

    if any(void_label):
        output[(mask_batch == void_label)] = void_label[0]
        output_old[(mask_batch == void_label).nonzero()] = void_label[0]

    pal = ['#%02x%02x%02x' % t for t in colors]
    sns.set_palette(pal)
    color_map = sns.color_palette()

    images = []
    for j in xrange(output.shape[0]):
        img = image_batch[j].transpose((1, 2, 0))  # / 255.
        label_out = my_label2rgb(output[j], colors=color_map)
        label_out_old = my_label2rgb(output_old[j], colors=color_map)
        mask_on_img = my_label2rgboverlay(mask_batch[j], colors=color_map,
                                          image=img, alpha=0.2)
        # pred_on_img = my_label2rgboverlay(output[j], colors=color_map,
        #                                   image=img, alpha=0.2)

        if img.shape[2] == 1:
            img = gray2rgb(img.squeeze())

        combined_image = np.concatenate((img, mask_on_img, label_out_old,
                                         label_out), axis=1)
        out_name = os.path.join(out_images_folder, tag + '_img' + str(j) +
                                '.png')
        scipy.misc.toimage(combined_image).save(out_name)
        images.append(combined_image)
    return images


def build_experiment_name(dae_dict, training_loss, data_aug, temperature=1.0):
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
    temperature: float
        Temperature to reduce the network confidence - default: 1.0 (does not
        change the output)
    """

    all_concat_h = '_'.join(dae_dict['concat_h'])
    all_loss = '_'.join(training_loss)

    exp_name = dae_dict['kind'] + '_' + all_concat_h

    if dae_dict['kind'] == 'standard':
        exp_name += '_f' + str(dae_dict['n_filters']) + 'c' + \
            str(dae_dict['conv_before_pool']) + 'p' + \
            str(dae_dict['additional_pool']) + \
            ('_skip' if dae_dict['skip'] else '')
        exp_name += '_' + dae_dict['unpool_type'] + \
                    ('_dropout' + str(dae_dict['dropout']) if
                    dae_dict['dropout'] > 0. else '')

    exp_name += '_' +  all_loss
    exp_name += ('_fromgt' if dae_dict['from_gt'] else '_fromfcn8') + '_z' + \
                str(dae_dict['noise'])
    exp_name += '_data_aug' if bool(data_aug) else ''
    exp_name += ('_T' + str(temperature)) if not dae_dict['from_gt'] else ''

    print(exp_name)

    return exp_name
