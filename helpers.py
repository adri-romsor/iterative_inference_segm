import os
import numpy as np
import scipy
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float
import seaborn as sns


def my_label2rgb(labels, colors):
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(colors)):
        output[(labels == i).nonzero()] = colors[i]
    return output


def my_label2rgboverlay(labels, colors, image, alpha=0.2):
    image_float = gray2rgb(img_as_float(rgb2gray(image) if
                                        image.shape[2] == 3 else
                                        np.squeeze(image)))
    label_image = my_label2rgb(labels, colors)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


def save_img(image_batch, mask_batch, output, output_old, out_images_folder,
             n_classes, tag, void_label, colors):

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


def build_experiment_name(dae_kind, layer_h, training_loss, from_gt, noise,
                          data_aug, temperature, n_filters, conv_before_pool,
                          additional_pool, skip, unpool_type, dropout):

    all_layer_h = '_'.join(layer_h)
    all_loss = '_'.join(training_loss)

    exp_name = dae_kind + '_' + all_layer_h

    if dae_kind == 'standard':
        exp_name += '_f' + str(n_filters) + 'c' + str(conv_before_pool) + \
            'p' + str(additional_pool) + ('_skip' if skip else '')
        exp_name += '_' + unpool_type + ('_dropout' + str(dropout) if
                                         dropout > 0. else '')

    exp_name += '_' +  all_loss
    exp_name += ('_fromgt' if from_gt else '_fromfcn8') + '_z' + str(noise)
    exp_name += '_data_aug' if data_aug else ''
    exp_name += ('_T' + str(temperature)) if not from_gt else ''

    print(exp_name)

    return exp_name
