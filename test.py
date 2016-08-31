import time
import sys
import numpy as np

batch_time = []
for j in range(5):

    for i in range(100):
        # Get minibatch
        start_time_batch = time.time()

        time.sleep(0.05)

        batch_time.append(time.time() - start_time_batch)
        estimated_time = (100-i+1)*np.mean(batch_time)
        progress = int(100*(i/100.))
        sys.stdout.write('\r [' + '#'*progress + '-'*(100 - progress) + '] Time left  {} '.format(estimated_time))
    print('\r coucou')