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

    output = output.argmax(1)
    output_old = output_old.argmax(1)

    if any(void_label):
        output[(mask_batch == void_label).nonzero()] = void_label[0]
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
