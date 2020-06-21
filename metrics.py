import numpy as np
# from sklearn.metrics import mean_squared_error
from math import sqrt
import json


# TODO: Handle varying sequence lengths


def _norm_factor(y, smoothing_const, norm_mode='range', reduction_axes=None):
    if reduction_axes is not None:
        reduction_axes = tuple(reduction_axes)
    if norm_mode == 'range':
        width = (np.max(y, axis=reduction_axes, keepdims=True)
                 - np.min(y, axis=reduction_axes, keepdims=True))
    elif norm_mode == 'mean':
        width = np.mean(y, axis=reduction_axes, keepdims=True)
    elif norm_mode == 'std':
        width = np.std(y, axis=reduction_axes, keepdims=True)
    elif norm_mode == 'quantile':
        width = (np.quantile(y, 0.75, axis=reduction_axes, keepdims=True)
                 - np.quantile(y, 0.25, axis=reduction_axes, keepdims=True))
    elif norm_mode == 'l2':
        width = np.linalg.norm(y, ord=2, axis=reduction_axes, keepdims=True)
    elif norm_mode == 'l1':
        width = np.linalg.norm(y, ord=1, axis=reduction_axes, keepdims=True)
    elif norm_mode == 'value':
        width = y
    else:
        raise ValueError('Unknown norm_mode = {:s}'.format(norm_mode))
    # Ensure no division by zero
    smooth_width = np.where(
        # np.abs(width) > smoothing_const, width, smoothing_const)
        np.abs(width) > smoothing_const, width, 1.0)
    return smooth_width


def mean_absolute_error(y, y_, reduction_axes=None):
    """Calculate the mean absolute error of two arrays.

    Args:
        y, y_:              Arrays
        reduction_axes:     The axes along which to reduce the arrays.

    """
    if reduction_axes is not None:
        reduction_axes = tuple(reduction_axes)
    d = y - y_
    ad = np.abs(d)
    mae = np.mean(ad, axis=reduction_axes)
    mae = np.squeeze(mae)
    if isinstance(mae, np.ndarray) and np.size(mae) == 1:
        return mae.item()
    else:
        return mae


def mean_absolute_percentage_error(
        y, y_, reduction_axes=None, norm_axes=None, norm_mode='value',
        smoothing_const=1e-8, percent=True):
    """Calculate the mean absolute percentage error of two arrays.

    Args:
        y, y_:              Arrays
        reduction_axes:     Axes along which to reduce the arrays
        norm_axes:          Axes along which to normalize the arrays
        norm_mode:          Normalization mode
        smoothing_const:    Smoothing constant to avoid division by zero
        percent:            Whether to scale up output values to percent
    """
    if reduction_axes is not None:
        reduction_axes = tuple(reduction_axes)
    if norm_axes is not None:
        norm_axes = tuple(norm_axes)
    else:
        norm_axes = reduction_axes
    n = _norm_factor(y, smoothing_const, norm_mode=norm_mode,
                     reduction_axes=norm_axes)
    r = (y / n) - (y_ / n)
    ar = np.abs(r)
    mre = np.mean(ar, axis=reduction_axes)
    mre = np.squeeze(mre)
    # Percentage
    if percent:
        mre *= 100.0
    if isinstance(mre, np.ndarray) and np.size(mre) == 1:
        return mre.item()
    else:
        return mre


def max_relative_error(
        y, y_, reduction_axes=None, norm_axes=None, norm_mode='value',
        smoothing_const=1e-8, percent=False):
    """Calculate the mean absolute percentage error of two arrays.

    Args:
        y, y_:              Arrays
        reduction_axes:     Axes along which to reduce the arrays
        norm_axes:          Axes along which to normalize the arrays
        norm_mode:          Normalization mode
        smoothing_const:    Smoothing constant to avoid division by zero
        percent:            Whether to scale up output values to percent
    """
    if reduction_axes is not None:
        reduction_axes = tuple(reduction_axes)
    if norm_axes is not None:
        norm_axes = tuple(norm_axes)
    else:
        norm_axes = reduction_axes
    n = _norm_factor(y, smoothing_const, norm_mode=norm_mode,
                     reduction_axes=norm_axes)
    r = (y / n) - (y_ / n)
    ar = np.abs(r)
    mre = np.max(ar, axis=reduction_axes)
    mre = np.squeeze(mre)
    # Percentage
    if percent:
        mre *= 100.0
    if isinstance(mre, np.ndarray) and np.size(mre) == 1:
        return mre.item()
    else:
        return mre


def symmetric_mean_absolute_percentage_error(
        y, y_, reduction_axes=None, norm_axes=None, norm_mode='value',
        smoothing_const=1e-8, percent=True):
    """Calculate the symmetric mean absolute percentage error of two arrays.

    Args:
        y, y_:              Arrays
        reduction_axes:     Axes along which to reduce the arrays
        norm_axes:          Axes along which to normalize the arrays
        norm_mode:          Normalization mode ('range', 'mean' or 'std)
        smoothing_const:    Smoothing constant to avoid division by zero
        percent:            Whether to scale up output values to percent
    """
    if reduction_axes is not None:
        reduction_axes = tuple(reduction_axes)
    if norm_axes is not None:
        norm_axes = tuple(norm_axes)
    else:
        norm_axes = reduction_axes
    n = _norm_factor(y, smoothing_const, norm_mode=norm_mode,
                     reduction_axes=norm_axes)
    n_ = _norm_factor(y_, smoothing_const, norm_mode=norm_mode,
                      reduction_axes=norm_axes)
    nn = (n + n_) / 2.0
    mr = (y / nn) - (y_ / nn)
    ar = np.abs(mr)
    mre = np.mean(ar, axis=reduction_axes)
    mre = np.squeeze(mre)
    # Percentage
    if percent:
        mre *= 100.0
    if isinstance(mre, np.ndarray) and np.size(mre) == 1:
        return mre.item()
    else:
        return mre


def root_mean_square_error(y, y_, reduction_axes=None):
    """Calculate the root mean square error of two arrays.

    Args:
        y, y_:              Arrays
        reduction_axes:     The axes along which to reduce the arrays.

    """
    if reduction_axes is not None:
        reduction_axes = tuple(reduction_axes)
    d = y - y_
    sd = np.square(d)
    mse = np.mean(sd, axis=reduction_axes)
    rmse = np.sqrt(mse)
    rmse = np.squeeze(rmse)
    if isinstance(rmse, np.ndarray) and np.size(rmse) == 1:
        return rmse.item()
    else:
        return np.squeeze(rmse)


def normalized_root_mean_square_error(
        y, y_, reduction_axes=None, norm_axes=None, norm_mode='range',
        smoothing_const=1e-8, percent=True):
    """Calculate the normalized root mean square error of two arrays.

    Args:
        y, y_:              Arrays
        reduction_axes:     Axes along which to reduce the arrays.
        norm_mode:          Normalization mode ('range', 'mean' or 'std)
        smoothing_const:    Smoothing constant to avoid division by zero
        percent:            Whether to scale up output values to percent
    """
    if reduction_axes is not None:
        reduction_axes = tuple(reduction_axes)
    if norm_axes is not None:
        norm_axes = tuple(norm_axes)
    else:
        norm_axes = reduction_axes
    # Calc root mean square error
    rmse = root_mean_square_error(y, y_, reduction_axes=norm_axes)
    # Normalize
    n = _norm_factor(y, smoothing_const, norm_mode=norm_mode,
                     reduction_axes=norm_axes)
    nrmse = rmse / n
    nrmse = np.mean(nrmse, axis=reduction_axes)
    nrmse = np.squeeze(nrmse)
    # Percentage
    if percent:
        nrmse *= 100.0
    if isinstance(nrmse, np.ndarray) and np.size(nrmse) == 1:
        return nrmse.item()
    else:
        return nrmse


def calc_rmse(x, y, reduction_axes=None):
    """
    Calculates the RMSE of the two arrays x and y along the reduction axes.
    If reduction_axes=None it is calculated over the whole arrays.
    :param x: np.array (ground truth)
    :param y: np.array (prediction)
    :param reduction_axes: tuple of axes along which the RMSE is calculated (e.g. (1,))
    :return: np.array of RMSE values. Shape of x without the reduction axes or ()
            in case of reduction_axes=None.
    """
    assert(np.shape(x) == np.shape(y))

    rmse = np.sqrt(np.square(x - y).mean(axis=reduction_axes))
    return np.mean(rmse), rmse


def calc_nrmse(x, y, norm_mode='range', percent=True, reduction_axes=None):
    """
    Calculates the normalized (relative) RMSE of the two arrays x and y
    along the reduction axes
    :param x: np.array (ground truth)
    :param y: np.array (prediction), same shape as x
    :param norm_mode: string specifying normalization by 'range' or 'mean'
    :param percent: boolean specifying output as percent or unit size
    :param reduction_axes: tuple of axes along which the NRMSE is calculated (e.g. (0, 1,))
    :return: np.array of NRMSE values. Shape of x without the reduction axes
    """
    assert (np.shape(x) == np.shape(y))
    if reduction_axes is not None:
        reduction_axes = tuple(reduction_axes)
    rmse = np.sqrt(np.square(x - y).mean(axis=reduction_axes))
    if norm_mode == 'mean':
        norm_factor = abs(np.mean(x, axis=reduction_axes))
    else:
        if norm_mode != 'range':
            print('Norm factor not implemented! Choose mean or range.')
            print('Normalizing by default (range)')
        norm_factor = (np.max(x, axis=reduction_axes)
                       - np.min(x, axis=reduction_axes))
    nrmse = rmse / np.max(
        [norm_factor, 1e-8 * np.ones_like(norm_factor)], axis=0)
    if percent:
        nrmse *= 100.0
    return np.mean(nrmse), nrmse


def calc_nrmse_mean(x, y, reduction_axes=None):
    return calc_nrmse(x, y, reduction_axes=reduction_axes, norm_mode='mean')


def calc_nrmse_range(x, y, reduction_axes=None):
    return calc_nrmse(x, y, reduction_axes=reduction_axes, norm_mode='range')


def calc_corr(x, y, reduction_axes=None):
    # TODO: generalize! Not just for 'NSF' and not just axes 0 or 0,1
    """
    NOT DONE!
    Calculates the Pearson correlation coefficient of the two arrays x and y
    along the reduction axes
    :param x: np.array (ground truth)
    :param y: np.array (prediction)
    :param reduction_axes: tuple of axes along which the corr is calculated (e.g. (0, 1,))
    :return: np.array of corr values. Shape of x without the reduction axes
    """
    assert(np.shape(x) == np.shape(y))

    if reduction_axes == (1,):
        corrs = np.zeros([np.size(x, 0), np.size(x, 2)])
        for k in range(np.size(x, 0)):
            for i in range(np.size(x, 2)):
                tmp = np.corrcoef(x[k, :, i].flatten(), y[k, :, i].flatten())
                corrs[k, i] = tmp[0, 1]
    elif reduction_axes == (0, 1,):
        corrs = np.zeros([np.size(x, 2)])
        for i in range(np.size(x, 2)):
            tmp = np.corrcoef(x[:, :, i].flatten(), y[:, :, i].flatten())
            corrs[i] = tmp[0, 1]
    else:
        raise NotImplementedError('error using reduction axes!')
    return np.mean(corrs), corrs


def _calc_corr(x, y, reduction_axes=None):
    assert (np.shape(x) == np.shape(y))

    def _corr(_x, _y):
        return np.corrcoef(_x, _y)[0, 1]

    return _calc_metric(x, y, _corr, reduction_axes=reduction_axes)


def _calc_metric(x, y, metric_fct, reduction_axes=None):
    """
    Calculates the specified metric for the arrays x and y regarding axes
    specified in reduction_axes
    :param x: [np.array] of ground truth
    :param y: [np.array] of approximations
    :param metric_fct: [callable] specify metric for eval
    (e.g. RMSE, NRMSE, Pearson correlation coefficient)
    :param reduction_axes: [tuple or None] specifies which axes of the arrays
    to calculate the metric for
    :return: (mean metric, metric array) of the results. Array shape according to
    the input arrays dimensions in reduction_axes
    """
    # TODO: Arrays of variable sequence lengths!

    if reduction_axes is not None:
        assert (np.ndim(x) > max(reduction_axes))
        reduction_axes = sorted(reduction_axes)
        if len(reduction_axes) == 1:
            slices_before_reduction = (slice(None),) * reduction_axes[0]
            metric_vec = np.zeros(np.size(y, reduction_axes[0]))
            for i in range(np.size(y, reduction_axes[0])):
                tmp_x = x[slices_before_reduction + (i,)]
                tmp_y = y[slices_before_reduction + (i,)]
                tmp_metric = metric_fct(tmp_x.flatten(), tmp_y.flatten())
                metric_vec[i] = tmp_metric
        elif len(reduction_axes) == 2:
            slices_before_reduction = (slice(None),) * reduction_axes[0]
            slices_after_reduction = (slice(None),) * (reduction_axes[1] - reduction_axes[0] - 1)
            metric_vec = np.zeros([np.size(y, reduction_axes[0]), np.size(y, reduction_axes[1])])
            for i in range(np.size(y, reduction_axes[0])):
                for j in range(np.size(y, reduction_axes[1])):
                    tmp_x = x[slices_before_reduction + (i,)
                              + slices_after_reduction + (j,)]
                    tmp_y = y[slices_before_reduction + (i,)
                              + slices_after_reduction + (j,)]
                    tmp_metric = metric_fct(tmp_x.flatten(), tmp_y.flatten())
                    metric_vec[i, j] = tmp_metric
        else:
            print('More than 2 reduction axes have not been implemented yet!!')
            return False
    else:
        metric_vec = np.array(sqrt(mean_squared_error(x.flatten(), y.flatten())))
    return np.mean(metric_vec), metric_vec
