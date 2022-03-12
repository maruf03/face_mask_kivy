import numpy as np
import cv2
import tensorflow as tf


def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """
    Helper function for SciPy argument validation.

    Many SciPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.

    Parameters
    ----------
    a : array_like
        The array-like input.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    sparse_ok : bool, optional
        True if scipy sparse matrices are allowed.
    objects_ok : bool, optional
        True if arrays with dype('O') are allowed.
    mask_ok : bool, optional
        True if masked arrays are allowed.
    as_inexact : bool, optional
        True to convert the input array to a np.inexact dtype.

    Returns
    -------
    ret : ndarray
        The converted validated array.

    """
    if not sparse_ok:
        import scipy.sparse
        if scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)
    if not mask_ok:
        if np.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    if not objects_ok:
        if a.dtype is np.dtype('O'):
            raise ValueError('object arrays are not supported')
    if as_inexact:
        if not np.issubdtype(a.dtype, np.inexact):
            a = toarray(a, dtype=np.float_)
    return a

def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.

        .. versionadded:: 0.11.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.

        .. versionadded:: 0.15.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.

        .. versionadded:: 0.12.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

        .. versionadded:: 0.16.0

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2

    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    Examples
    --------
    >>> from scipy.special import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107

    With weights

    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647

    Returning a sign flag

    >>> logsumexp([1,2],b=[1,-1],return_sign=True)
    (1.5413248546129181, -1.0)

    Notice that `logsumexp` does not directly support masked arrays. To use it
    on a masked array, convert the mask into zero weights:

    >>> a = np.ma.array([np.log(2), 2, np.log(3)],
    ...                  mask=[False, True, False])
    >>> b = (~a.mask).astype(int)
    >>> logsumexp(a.data, b=b), np.log(5)
    1.6094379124341005, 1.6094379124341005

    """
    a = _asarray_validated(a, check_finite=False)
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out

def softmax(x, axis=None):
    r"""
    Softmax function

    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::

        softmax(x) = np.exp(x)/sum(np.exp(x))

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.

    Notes
    -----
    The formula for the softmax function :math:`\sigma(x)` for a vector
    :math:`x = \{x_0, x_1, ..., x_{n-1}\}` is

    .. math:: \sigma(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}

    The `softmax` function is the gradient of `logsumexp`.

    .. versionadded:: 1.2.0

    Examples
    --------
    >>> from scipy.special import softmax
    >>> np.set_printoptions(precision=5)

    >>> x = np.array([[1, 0.5, 0.2, 3],
    ...               [1,  -1,   7, 3],
    ...               [2,  12,  13, 3]])
    ...

    Compute the softmax transformation over the entire array.

    >>> m = softmax(x)
    >>> m
    array([[  4.48309e-06,   2.71913e-06,   2.01438e-06,   3.31258e-05],
           [  4.48309e-06,   6.06720e-07,   1.80861e-03,   3.31258e-05],
           [  1.21863e-05,   2.68421e-01,   7.29644e-01,   3.31258e-05]])

    >>> m.sum()
    1.0000000000000002

    Compute the softmax transformation along the first axis (i.e., the
    columns).

    >>> m = softmax(x, axis=0)

    >>> m
    array([[  2.11942e-01,   1.01300e-05,   2.75394e-06,   3.33333e-01],
           [  2.11942e-01,   2.26030e-06,   2.47262e-03,   3.33333e-01],
           [  5.76117e-01,   9.99988e-01,   9.97525e-01,   3.33333e-01]])

    >>> m.sum(axis=0)
    array([ 1.,  1.,  1.,  1.])

    Compute the softmax transformation along the second axis (i.e., the rows).

    >>> m = softmax(x, axis=1)
    >>> m
    array([[  1.05877e-01,   6.42177e-02,   4.75736e-02,   7.82332e-01],
           [  2.42746e-03,   3.28521e-04,   9.79307e-01,   1.79366e-02],
           [  1.22094e-05,   2.68929e-01,   7.31025e-01,   3.31885e-05]])

    >>> m.sum(axis=1)
    array([ 1.,  1.,  1.])

    """

    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))
#https://github.com/tito/experiment-tensorflow-lite


# face detection model
face_detection_model = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt',
                                                './models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
# face mask recognition model
model = tf.keras.models.load_model('face_cnn_model/')
# interpreter = tf.lite.Interpreter(model_path='model.tflite')
# interpreter.allocate_tensors()

# label
labels = ['Mask', 'No Mask', 'Covered Mouth Chin', 'Covered Nose Mouth']


def getColor(label):
    if label == "Mask":
        color = (0, 255, 0)

    elif label == 'No Mask':
        color = (0, 0, 255)
    elif label == 'Covered Mouth Chin':
        color = (0, 255, 255)
    else:
        color = (255, 255, 0)

    return color


def face_mask_prediction(img):
    # step - 1 : face detection
    image = img.copy()
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1, (300, 300), (104, 117, 123), swapRB=True)
    #
    face_detection_model.setInput(blob)
    detection = face_detection_model.forward()  # it will give the detection
    for i in range(0, detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.5:
            box = detection[0, 0, i, 3:7]*np.array([w, h, w, h])
            box = box.astype(int)
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            # cv2.rectangle(image,pt1,pt2,(0,255,0),1)

            # step -2: Data preprocessing
            face = image[box[1]:box[3], box[0]:box[2]]
            face_blob = cv2.dnn.blobFromImage(
                face, 1, (100, 100), (104, 117, 123), swapRB=True)
            face_blob_squeeze = np.squeeze(face_blob).T
            face_blob_rotate = cv2.rotate(
                face_blob_squeeze, cv2.ROTATE_90_CLOCKWISE)
            face_blob_flip = cv2.flip(face_blob_rotate, 1)
            # normalization
            img_norm = np.maximum(face_blob_flip, 0)/face_blob_flip.max()
            # step-3: Deep Learning (CNN)
            img_input = img_norm.reshape(1, 100, 100, 3)
            result = model.predict(img_input)
            result = softmax(result)[0]

            confidence_index = result.argmax()
            confidence_score = result[confidence_index]
            label = labels[confidence_index]
            label_text = '{}: {:,.0f} %'.format(label, confidence_score*100)
            #print(label_text)
            # color
            color = getColor(label)
            cv2.rectangle(image, pt1, pt2, color, 1)
            cv2.putText(image, label_text, pt1,
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    return image
