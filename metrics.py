import theano.tensor as T
from theano import config

_FLOATX = config.floatX
_EPSILON = 10e-8


def jaccard(y_pred, y_true, n_classes):

    # y_pred to indices
    if T.gt(y_pred.shape[1], 1):
        y_pred = T.argmax(y_pred, axis=1)
    else:
        y_pred = T.flatten(y_pred)
        y_pred = T.set_subtensor(y_pred[T.ge(y_pred, 0.5).nonzero()], 1)
        y_pred = T.set_subtensor(y_pred[T.lt(y_pred, 0.5).nonzero()], 0)

    # Compute confusion matrix
    cm = T.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cm = T.set_subtensor(
                cm[i, j], T.sum(T.eq(y_pred, i) * T.eq(y_true, j)))

    # Compute Jaccard Index
    TP_perclass = T.cast(cm.diagonal(), _FLOATX)
    FP_perclass = cm.sum(1) - TP_perclass
    FN_perclass = cm.sum(0) - TP_perclass

    num = TP_perclass
    denom = TP_perclass + FP_perclass + FN_perclass

    return T.stack([num, denom], axis=0)


def accuracy(y_pred, y_true, void_labels):

    # y_pred to indices
    if T.gt(y_pred.shape[1], 1):
        y_pred = T.argmax(y_pred, axis=1)
    else:
        pass
        y_pred = T.flatten(y_pred)
        y_pred = T.set_subtensor(y_pred[T.ge(y_pred, 0.5).nonzero()], 1)
        y_pred = T.set_subtensor(y_pred[T.lt(y_pred, 0.5).nonzero()], 0)

    # Compute accuracy
    acc = T.eq(y_pred, y_true)

    # Create mask
    mask = T.ones_like(y_true)
    for el in void_labels:
        indices = T.eq(y_true, el).nonzero()
        if any(indices):
            mask = T.set_subtensor(mask[indices], 0.)

    # Apply mask
    acc *= mask
    acc = T.sum(acc) / T.sum(mask).astype(_FLOATX)

    return acc


def crossentropy(y_pred, y_true, void_labels):
    # Clip predictions
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

    # Create mask
    mask = T.ones_like(y_true)
    for el in void_labels:
        mask = T.set_subtensor(mask[T.eq(y_true, el).nonzero()], 0.)

    # Modify y_true temporarily
    y_true_tmp = y_true * mask

    # Compute cross-entropy
    loss = T.nnet.categorical_crossentropy(y_pred, y_true_tmp)

    # Compute masked mean loss
    loss *= mask
    loss = T.sum(loss) / T.sum(mask).astype(_FLOATX)

    return loss


def binary_crossentropy(y_pred, y_true):
    # Clip predictions to avoid numerical instability
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

    loss = T.nnet.binary_crossentropy(y_pred, y_true)

    return loss.mean()
