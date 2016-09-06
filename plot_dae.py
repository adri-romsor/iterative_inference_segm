import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from getpass import getuser

if getuser() == 'romerosa':
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
elif getuser() == 'jegousim':
    LOADPATH = '/data/lisatmp4/jegousim/iterative_inference/'
else:
    raise ValueError('Unknown user : {}'.format(getuser()))


def plot(dataset,
         model_path=None,
         models=None,
         colors=None):

    for (m, c) in zip(models, colors):
        file_path = os.path.join(model_path, dataset, m)
        if not os.path.exists(file_path):
            raise ValueError('The path to {} does not exist'.format(file_path))

        error_var = np.load(os.path.join(file_path, 'dae_errors.npz'))
        train_err = error_var['arr_1']
        val_err = error_var['arr_0']

        max_epoch = len(train_err)
        epochs = range(max_epoch)

        plt.plot(epochs, train_err, '-'+c, label='train: '+m)
        plt.plot(epochs, val_err, '--'+c, label='val: '+m)

    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='plot errors')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset')
    parser.add_argument('-path',
                        default='/data/lisatmp4/romerosa/itinf/models/',
                        help='Path to errors file')
    parser.add_argument('-models',
                        type=list,
                        default=[# 'pool3_f64c1p2_z0.1_squared_error_fromfcn8_standard',
                                 'pool3_f64c1p2_z0.1_crossentropy_fromfcn8_standard',
                                 'fcn8',
                                 # 'pool3_f64c1p2_z0.1_squared_error_fromgt_standard',
                                 # 'pool3_f64c1p2_z0.1_squared_error_skip_fromfcn8_standard',
                                 # 'pool3_f64c1p2_z0.1_squared_error_skip_fromgt_standard',
                                 # 'input_f64c1p5_z0.1_squared_error_fromgt_standard',
                                 # 'input_f64c1p5_z0.1_squared_error_skip_fromgt_standard',
                                 # 'pool1_f64c1p4_z0.1_squared_error_fromgt_standard'
                                 # 'pool4_f64c1p1_z0.1_squared_error_fromgt_standard'
                                ],
                        help='List of model names.')
    parser.add_argument('-colors',
                        type=list,
                        default=['r', 'g', 'b', 'k', 'c', 'm', 'y'],
                        help='Colors to plot curves.')
    args = parser.parse_args()

    plot(args.dataset, args.path, args.models, args.colors)

if __name__ == "__main__":
    main()
