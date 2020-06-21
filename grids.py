from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd
import warnings
import copy

from sortedcontainers import SortedDict
from scipy import ndimage
from matplotlib.path import Path


DEBUG = False


def generate_grid(geometry, grid_shape):
    """Return the coordinates of the internal grid."""
    for g in grid_shape:
        assert g % 2, 'grid shapes must be uneven!'
    axes = [np.linspace(0.0, l, g) for l, g in zip(geometry, grid_shape)]
    # Generate meshgrid
    meshgrid = np.asarray(np.meshgrid(*axes, indexing='ij'))
    # Change ordering from (component, axis0, axis1, axis2) to
    #    (axis0, axis1, axis2, component)
    order = list(range(0, meshgrid.ndim))
    order.reverse()
    meshgrid = np.transpose(meshgrid, order)
    # Reshape to size (num_nodes, num_dims)
    coords = np.reshape(meshgrid, [-1, len(grid_shape)])
    return coords


def extract_boundary_grid_labels(grid_labels, grid_shape):
    boundary_mask = np.zeros(np.asarray(grid_shape) - 2, dtype=np.bool)
    boundary_mask = np.pad(boundary_mask, 1, mode='constant',
                           constant_values=1)
    # Ensure boundary grid labels grow along x axis first
    return grid_labels.T[boundary_mask.T]


def calc_grid_labels(grid_shape):
    return np.reshape(np.arange(1, np.prod(grid_shape) + 1), grid_shape,
                      order='F')


def sort_nodes_positive_rotation(node_coords, grid_shape):
    num_x = grid_shape[0]
    num_y = grid_shape[1]
    #  Sort nodes in positive mathematical rotation
    dx = num_x - 2
    dy = num_y - 2
    idx_outer = (list(range(0, dx + 2))
                 + list(range(dx + 3, dx + 2 * dy + 2, 2))
                 + list(range(num_x * num_y - dx * dy - 1,
                              dx + 2 * dy + 1,
                              -1))
                 + list(range(dx + 2 * dy - 2, dx + 1, -2)))
    return node_coords[idx_outer, :]


def sort_grid_boundary_positive_rotation(boundary_coords, grid_shape,
                                         is_cube=True):
    if is_cube:
        num_x = grid_shape[0]
        num_y = grid_shape[1]
        #  Sort nodes in positive mathematical rotation
        dx = num_x - 2
        dy = num_y - 2
        idx_outer = (list(range(0, dy + 2))
                     + list(range(dy + 3, dy + 2 * dx + 2, 2))
                     + list(range(num_y * num_x - dy * dx - 1,
                                  dy + 2 * dx + 1,
                                  -1))
                     + list(range(dy + 2 * dx - 2, dy + 1, -2)))
    else:
        # Brute force search algorithm
        idx_outer = np.zeros(len(boundary_coords), dtype=np.uint32)
        for i in range(1, len(boundary_coords)):
            current = boundary_coords[idx_outer[i - 1]]
            distance = np.linalg.norm(boundary_coords - current, axis=1)
            distance[idx_outer[i - 1]] = np.finfo(distance.dtype).max
            next0 = np.argmin(distance)
            distance[next0] = np.finfo(distance.dtype).max
            next1 = np.argmin(distance)
            if not (next0 in idx_outer or next1 in idx_outer):
                # Pick value with larger x value
                if (boundary_coords[next0, 0]
                        >= boundary_coords[next1, 0]):
                    idx_outer[i] = next0
                else:
                    idx_outer[i] = next1
            elif next0 in idx_outer and not next1 in idx_outer:
                idx_outer[i] = next1
            elif next1 in idx_outer and not next0 in idx_outer:
                idx_outer[i] = next0
            else:
                assert len(idx_outer) == len(np.unique(idx_outer))
                break
    return idx_outer


def repair_voids(body, grid_data):
    if isinstance(grid_data, (list, tuple)):
        grid_dict = SortedDict({i: d for i, d in enumerate(grid_data)})
    else:
        grid_dict = grid_data
    grid_shape = list(np.shape(body)[:-1])
    num_grid_dims = np.size(grid_shape)
    fixed_dict = copy.deepcopy(grid_dict)
    # Filter out isolated small voids
    filtered_body = np.copy(body)
    if num_grid_dims == 2:
        body_slice = body[..., 0]
        filtered_body_slice = np.copy(body_slice)
        id_regions, num_ids = ndimage.label(
            np.logical_not(body_slice))
        id_sizes = np.asarray(ndimage.sum(
            np.logical_not(body_slice), id_regions, range(num_ids + 1)))
        area_mask = (id_sizes == 1)
        filtered_body_slice[area_mask[id_regions]] = 1.0
        # Average feature values between neighbouring cells for voids
        filtered_out = np.logical_and(np.logical_not(body_slice),
                                      filtered_body_slice)
        ii, jj = filtered_out.nonzero()
        for i, j in zip(ii, jj):
            im = max(0, i - 1)
            ip = min(grid_shape[0], i + 1) + 1
            jm = max(0, j - 1)
            jp = min(grid_shape[1], j + 1) + 1
            for k, V in fixed_dict.items():
                neighborhood = V[im:ip, jm:jp, :]
                if np.any(neighborhood != 0.0):
                    fixed_dict[k][i, j, :] = (np.sum(neighborhood, axis=(0, 1))
                                              / np.sum(body[im:ip, jm:jp]))
        # Assign
        body[..., 0] = filtered_body_slice
    elif num_grid_dims == 3:
        for k in range(filtered_body.shape[-2]):  # -2: last spatial axis
            body_slice = body[..., k, 0]
            filtered_body_slice = np.copy(body_slice)
            id_regions, num_ids = ndimage.label(
                np.logical_not(body_slice))
            id_sizes = np.asarray(ndimage.sum(
                np.logical_not(body_slice), id_regions, range(num_ids + 1)))
            area_mask = (id_sizes == 1)
            filtered_body_slice[area_mask[id_regions]] = 1.0
            # Average feature values between neighbouring cells for voids
            filtered_out = np.logical_and(np.logical_not(body_slice),
                                          filtered_body_slice)
            ii, jj = filtered_out.nonzero()
            km = max(0, k - 1)
            kp = min(grid_shape[2], k + 1) + 1
            for i, j in zip(ii, jj):
                im = max(0, i - 1)
                ip = min(grid_shape[0], i + 1) + 1
                jm = max(0, j - 1)
                jp = min(grid_shape[1], j + 1) + 1
                for k, V in fixed_dict.items():
                    neighborhood = V[im:ip, jm:jp, km:kp, :]
                    if np.any(neighborhood != 0.0):
                        fixed_dict[k][i, j, k, :] = (
                                    np.sum(neighborhood, axis=(0, 1, 2))
                                    / np.sum(body[im:ip, jm:jp, km:kp]))
            # Assign
            body[..., k, 0] = filtered_body_slice
    else:
        raise NotImplementedError
    # Return to original data type
    if isinstance(grid_data, (list, tuple)):
        return body, type(grid_data)([v for k, v in grid_dict.items()])
    else:
        return body, grid_dict


def find_boundary(body):
    grid_shape = list(np.shape(body)[:-1])
    num_grid_dims = len(grid_shape)
    # Find edges
    boundary = np.zeros(grid_shape + [1], dtype=np.float64)
    for d in range(num_grid_dims):
        boundary = np.hypot(ndimage.sobel(body, axis=d), boundary)
        boundary[boundary > 0.5] = 1.0
        boundary[boundary <= 0.5] = 0.0
    boundary[boundary > 0.5] = 1.0
    boundary[boundary <= 0.5] = 0.0
    boundary[0, ..., 0] = 1.0  # Set left boundary to 1
    boundary[-1, ..., 0] = 1.0  # Set right boundary to 1
    if num_grid_dims > 2:
        boundary[:, 0, :, 0] = 1.0  # Set front boundary to 1
        boundary[:, -1, :, 0] = 1.0  # Set back boundary to 1
    boundary[..., 0, 0] = 1.0  # Set bottom boundary to 1
    boundary[..., -1, 0] = 1.0  # Set top boundary to 1
    boundary *= body  # Mask part of edges that is not in body
    return boundary


def grid_encode(grid_coords, grid_shape, point_coords, point_data):
    """Sort point coordinates and point data into rectangular grid."""
    if isinstance(point_data, (list, tuple)):
        point_dict = SortedDict({i: d for i, d in enumerate(point_data)})
    else:
        point_dict = point_data
    grid_shape = list(grid_shape)
    num_grid_dims = len(grid_shape)
    # Center each voxel bin at internal grid positions
    bin_centers = [np.unique(grid_coords[:, i])
                   for i in range(num_grid_dims)]
    bin_widths = [np.diff(b) for b in bin_centers]
    bins = [np.concatenate([c[0:1] - w[0:1] / 2.0,
                            c[:-1] + w / 2.0,
                            c[-1:] + w[-1:] / 2.0])
            for c, w in zip(bin_centers, bin_widths)]
    # Get indices of nodes belonging to which bin
    idx = [pd.cut(point_coords[:, i], bins[i], right=True,
                  include_lowest=False).codes
           for i in range(num_grid_dims)]
    # Initialize variables
    X = np.zeros(grid_shape + [point_coords.shape[-1]],
                 dtype=np.float64)
    grid_dict = {
        k: (np.zeros(grid_shape + [v.shape[-1]], dtype=np.float64) if v.shape
            else np.zeros(grid_shape + [1], dtype=np.float64))
        for k, v in point_dict.items()}
    body = np.zeros(grid_shape + [1], dtype=np.float64)
    # Sort values into grid
    X[tuple(idx + [slice(None)])] = point_coords
    for k in grid_dict.keys():
        try:
            grid_dict[k][tuple(idx + [slice(None)])] = point_dict[k]
        except ValueError:
            if DEBUG:
                warnings.warn(
                    'Array for key "{:s}" could not be encoded into grid.'
                        .format(k))
    body[tuple(idx + [0])] = 1.0
    # Return to original data type
    if isinstance(point_data, (list, tuple)):
        return X, body, type(point_data)([v for k, v in grid_dict.items()])
    else:
        return X, body, grid_dict


def assemble_path(coords):
    coords = np.asarray(coords)
    verts = np.concatenate([coords, coords[0:1, :]], axis=0)
    codes = tuple([Path.MOVETO] + [Path.LINETO] * (verts.shape[0] - 2)
                  + [Path.CLOSEPOLY])
    return Path(verts, codes=codes)
