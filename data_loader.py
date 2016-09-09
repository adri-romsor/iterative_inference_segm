from dataset_loaders.videos.colonoscopyVideos import PolypVideoDataset
from dataset_loaders.images.camvid import CamvidDataset
from dataset_loaders.images.em_stacks import IsbiEmStacksDataset


def load_data(dataset, train_crop_size=(224, 224), one_hot=False,
              batch_size=[10, 10, 10],
              rotation_range=0.,
              width_shift_range=0.,
              height_shift_range=0.,
              shear_range=0.,
              zoom_range=0.,
              channel_shift_range=0.,
              fill_mode='nearest',
              cval=0.,
              cvalMask=0.,
              horizontal_flip=False,
              vertical_flip=False,
              rescale=None,
              spline_warp=False,
              warp_sigma=0.1,
              warp_grid_size=3,
              elastic_def=False):

    # Build dataset iterator
    if dataset == 'polyp_videos':
        train_iter = PolypVideoDataset(which_set='train',
                                       batch_size=batch_size[0],
                                       seq_per_video=0,
                                       seq_length=0,
                                       crop_size=train_crop_size,
                                       rotation_range=rotation_range,
                                       width_shift_range=width_shift_range,
                                       height_shift_range=height_shift_range,
                                       shear_range=shear_range,
                                       zoom_range=zoom_range,
                                       channel_shift_range=channel_shift_range,
                                       fill_mode=fill_mode,
                                       cval=cval,
                                       cvalMask=cvalMask,
                                       horizontal_flip=horizontal_flip,
                                       vertical_flip=vertical_flip,
                                       rescale=rescale,
                                       spline_warp=spline_warp,
                                       warp_sigma=warp_sigma,
                                       warp_grid_size=warp_grid_size,
                                       split=.75,
                                       get_one_hot=one_hot,
                                       get_01c=False,
                                       overlap=0,
                                       use_threads=True)
        val_iter = PolypVideoDataset(which_set='val',
                                     batch_size=batch_size[1],
                                     seq_per_video=0,
                                     seq_length=0,
                                     crop_size=None,
                                     split=.75,
                                     get_one_hot=one_hot,
                                     get_01c=False,
                                     overlap=0,
                                     use_threads=True)
        test_iter = PolypVideoDataset(which_set='test',
                                      batch_size=batch_size[2],
                                      seq_per_video=0,
                                      seq_length=0,
                                      crop_size=None,
                                      get_one_hot=one_hot,
                                      get_01c=False,
                                      overlap=0,
                                      use_threads=True)

    elif dataset == 'camvid':
        train_iter = CamvidDataset(which_set='train',
                                   batch_size=batch_size[0],
                                   seq_per_video=0,
                                   seq_length=0,
                                   crop_size=train_crop_size,
                                   rotation_range=rotation_range,
                                   width_shift_range=width_shift_range,
                                   height_shift_range=height_shift_range,
                                   shear_range=shear_range,
                                   zoom_range=zoom_range,
                                   channel_shift_range=channel_shift_range,
                                   fill_mode=fill_mode,
                                   cval=cval,
                                   cvalMask=cvalMask,
                                   horizontal_flip=horizontal_flip,
                                   vertical_flip=vertical_flip,
                                   rescale=rescale,
                                   spline_warp=spline_warp,
                                   warp_sigma=warp_sigma,
                                   warp_grid_size=warp_grid_size,
                                   get_one_hot=one_hot,
                                   get_01c=False,
                                   overlap=0,
                                   use_threads=True)
        val_iter = CamvidDataset(which_set='val',
                                 batch_size=batch_size[1],
                                 seq_per_video=0,
                                 seq_length=0,
                                 crop_size=None,
                                 get_one_hot=one_hot,
                                 get_01c=False,
                                 overlap=0,
                                 use_threads=True)
        test_iter = CamvidDataset(which_set='test',
                                  batch_size=batch_size[2],
                                  seq_per_video=0,
                                  seq_length=0,
                                  crop_size=None,
                                  get_one_hot=one_hot,
                                  get_01c=False,
                                  overlap=0,
                                  use_threads=True)
    elif dataset == 'em':
        train_iter = IsbiEmStacksDataset(which_set='train',
                                         start=0,
                                         end=25,
                                         batch_size=batch_size[0],
                                         seq_per_video=0,
                                         seq_length=0,
                                         crop_size=train_crop_size,
                                         rotation_range=rotation_range,
                                         width_shift_range=width_shift_range,
                                         height_shift_range=height_shift_range,
                                         shear_range=shear_range,
                                         zoom_range=zoom_range,
                                         channel_shift_range=channel_shift_range,
                                         fill_mode=fill_mode,
                                         cval=cval,
                                         cvalMask=cvalMask,
                                         horizontal_flip=horizontal_flip,
                                         vertical_flip=vertical_flip,
                                         rescale=rescale,
                                         spline_warp=spline_warp,
                                         warp_sigma=warp_sigma,
                                         warp_grid_size=warp_grid_size,
                                         get_one_hot=one_hot,
                                         get_01c=False,
                                         overlap=0,
                                         use_threads=True,
                                         elastic_deform=elastic_def)

        val_iter = IsbiEmStacksDataset(which_set='train',
                                       batch_size=batch_size[1],
                                       seq_per_video=0,
                                       seq_length=0,
                                       crop_size=None,
                                       get_one_hot=False,
                                       get_01c=False,
                                       use_threads=True,
                                       shuffle_at_each_epoch=False,
                                       start=26,
                                       end=30)
        test_iter = None
    else:
        raise NotImplementedError

    return train_iter, val_iter, test_iter
