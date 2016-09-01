import time
import sys
import numpy as np

n_batches_train = 10
for epoch in range(5):
    # Single epoch training and validation
    start_time = time.time()

    # Train

    batch_time = []

    for i in range(n_batches_train):
        start_time_batch = time.time()
        time.sleep(0.1)

        if epoch == 0:
            batch_time.append(time.time() - start_time_batch)
            mean_batch_time = np.mean(batch_time)

        remaining_time = int((n_batches_train - i + 1) * mean_batch_time)
        progression = int(50. * ((i + 1) / float(n_batches_train)))

        sys.stdout.write("\r")
        progbar = 'Remaining time = {} sec. Cost train = {}' \
            .format(remaining_time, i / (i + 1))
        sys.stdout.write(progbar)
        sys.stdout.flush()

    print('\r \x1b[2 Epoch {} finished'.format(epoch))
    sys.stdout.flush()
