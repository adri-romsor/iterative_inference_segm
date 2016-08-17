import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot(error_path='/data/lisatmp4/romerosa/itinf/errors/',
         error_file=['errors_earlyjacc.npz'],
         colors=None):

    for (f, c) in zip(error_file, colors):
        error_var = np.load(error_path + f)
        train_err = error_var['arr_1']
        val_err = error_var['arr_0']

        max_epoch = len(train_err)
        epochs = range(max_epoch)

        plt.plot(epochs, train_err, '-'+c, label='train: '+f[:-4])
        plt.plot(epochs, val_err, '--'+c, label='val: '+f[:-4])

    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='plot errors')
    parser.add_argument('-path',
                        default='/data/lisatmp4/romerosa/itinf/errors/',
                        help='Path to errors file')
    parser.add_argument('-file',
                        type=list,
                        default=['dae_errors_pool1.npz',
                                 'dae_errors_pool3.npz'],
                        help='Errors file.')
    parser.add_argument('-colors',
                        type=list,
                        default=['r', 'g'],
                        help='Errors file.')
    args = parser.parse_args()

    plot(args.path, args.file, args.colors)

if __name__ == "__main__":
    main()
