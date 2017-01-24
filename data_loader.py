from dataset_loaders.images.polyps912 import Polyps912Dataset
from dataset_loaders.images.camvid import CamvidDataset
from dataset_loaders.images.polyps912 import Polyps912Dataset
from dataset_loaders.images.isbi_em_stacks import IsbiEmStacksDataset


def load_data(dataset, train_data_augm_kwargs={}, one_hot=False,
              batch_size=[10, 10, 10], shuffle_train=True):

    # Build dataset iterator
    if dataset == 'polyps912':
        train_iter = Polyps912Dataset(which_set='train',
                                      batch_size=batch_size[0],
                                      seq_per_subset=0,
                                      seq_length=0,
                                      data_augm_kwargs=train_data_augm_kwargs,
                                      return_one_hot=one_hot,
                                      return_01c=False,
                                      overlap=0,
                                      use_threads=False,
                                      shuffle_at_each_epoch=shuffle_train,
                                      return_list=True)
        val_iter = Polyps912Dataset(which_set='val',
                                    batch_size=batch_size[1],
                                    seq_per_subset=0,
                                    seq_length=0,
                                    return_one_hot=one_hot,
                                    return_01c=False,
                                    overlap=0,
                                    use_threads=False,
                                    shuffle_at_each_epoch=False,
                                    return_list=True)
        test_iter = Polyps912Dataset(which_set='test',
                                     batch_size=batch_size[2],
                                     seq_per_subset=0,
                                     seq_length=0,
                                     return_one_hot=one_hot,
                                     return_01c=False,
                                     overlap=0,
                                     use_threads=False,
                                     shuffle_at_each_epoch=False,
                                     return_list=True)
    elif dataset == 'camvid':
        train_iter = CamvidDataset(which_set='train',
                                   batch_size=batch_size[0],
                                   seq_per_subset=0,
                                   seq_length=0,
                                   data_augm_kwargs=train_data_augm_kwargs,
                                   return_one_hot=one_hot,
                                   return_01c=False,
                                   overlap=0,
                                   use_threads=True,
                                   shuffle_at_each_epoch=shuffle_train,
                                   return_list=True)
        val_iter = CamvidDataset(which_set='val',
                                 batch_size=batch_size[1],
                                 seq_per_subset=0,
                                 seq_length=0,
                                 return_one_hot=one_hot,
                                 return_01c=False,
                                 overlap=0,
                                 use_threads=True,
                                 shuffle_at_each_epoch=False,
                                 return_list=True)
        test_iter = CamvidDataset(which_set='test',
                                  batch_size=batch_size[2],
                                  seq_per_subset=0,
                                  seq_length=0,
                                  return_one_hot=one_hot,
                                  return_01c=False,
                                  overlap=0,
                                  use_threads=True,
                                  shuffle_at_each_epoch=False,
                                  return_list=True)
    elif dataset == 'em':
        train_iter = IsbiEmStacksDataset(which_set='train',
                                         start=0,
                                         end=25,
                                         batch_size=batch_size[0],
                                         seq_per_subset=0,
                                         seq_length=0,
                                         data_augm_kwargs=train_data_augm_kwargs,
                                         return_one_hot=one_hot,
                                         return_01c=False,
                                         overlap=0,
                                         use_threads=True,
                                         shuffle_at_each_epoch=shuffle_train,
                                         return_list=True)

        val_iter = IsbiEmStacksDataset(which_set='train',
                                       batch_size=batch_size[1],
                                       seq_per_subset=0,
                                       seq_length=0,
                                       return_one_hot=one_hot,
                                       return_01c=False,
                                       use_threads=True,
                                       shuffle_at_each_epoch=False,
                                       start=25,
                                       end=30,
                                       return_list=True)
        test_iter = None
    elif dataset == 'polyps912':
        train_iter = Polyps912Dataset(which_set='train',
                                      batch_size=10,
                                      seq_per_subset=0,
                                      seq_length=0,
                                      data_augm_kwargs=train_data_augm_kwargs,
                                      return_one_hot=True,
                                      return_01c=True,
                                      use_threads=False,
                                      return_list=True)
        val_iter = Polyps912Dataset(which_set='val',
                                    batch_size=10,
                                    seq_per_subset=0,
                                    seq_length=0,
                                    return_one_hot=True,
                                    return_01c=True,
                                    use_threads=False,
                                    return_list=True)
        test_iter = Polyps912Dataset(which_set='test',
                                     batch_size=10,
                                     seq_per_subset=0,
                                     seq_length=0,
                                     return_one_hot=True,
                                     return_01c=True,
                                     use_threads=False,
                                     return_list=True)
    else:
        raise NotImplementedError

    return train_iter, val_iter, test_iter
