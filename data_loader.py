from dataset_loaders.videos.colonoscopyVideos import PolypVideoDataset
from dataset_loaders.images.camvid import CamvidDataset


def load_data(dataset, train_crop_size=(224, 224), one_hot=False):

    # Build dataset iterator
    if dataset == 'polyp_videos':
        train_iter = PolypVideoDataset(which_set='train',
                                       batch_size=10,
                                       seq_per_video=0,
                                       seq_length=0,
                                       crop_size=train_crop_size,
                                       split=.75,
                                       get_one_hot=one_hot,
                                       get_01c=False,
                                       use_threads=True)
        val_iter = PolypVideoDataset(which_set='val',
                                     batch_size=1,
                                     seq_per_video=0,
                                     seq_length=0,
                                     crop_size=None,
                                     split=.75,
                                     get_one_hot=one_hot,
                                     get_01c=False,
                                     use_threads=True)
        test_iter = PolypVideoDataset(which_set='test',
                                      batch_size=1,
                                      seq_per_video=0,
                                      seq_length=0,
                                      crop_size=None,
                                      get_one_hot=one_hot,
                                      get_01c=False,
                                      use_threads=True)

    elif dataset == 'camvid':
        train_iter = CamvidDataset(which_set='train',
                                   batch_size=10,
                                   seq_per_video=0,
                                   seq_length=0,
                                   crop_size=train_crop_size,
                                   get_one_hot=one_hot,
                                   get_01c=False,
                                   use_threads=True)
        val_iter = CamvidDataset(which_set='val',
                                 batch_size=10,
                                 seq_per_video=0,
                                 seq_length=0,
                                 crop_size=None,
                                 get_one_hot=one_hot,
                                 get_01c=False,
                                 use_threads=True)
        test_iter = CamvidDataset(which_set='test',
                                  batch_size=10,
                                  seq_per_video=0,
                                  seq_length=0,
                                  crop_size=None,
                                  get_one_hot=one_hot,
                                  get_01c=False,
                                  use_threads=True)
    else:
        raise NotImplementedError

    return train_iter, val_iter, test_iter
