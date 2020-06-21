"""CIDS computational intelligence library"""

# COMPATIBILITY IMPORT
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

# IMPORT
import os
import warnings
import tensorflow as tf
import numpy as np
import matplotlib


from sortedcontainers import SortedDict
from collections import OrderedDict
from grids import *
from numpy import ndarray
from copy import deepcopy
from warnings import warn
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def read_axes(data_format):
    """Returns the axe indices from the data format string."""
    assert len(data_format) > 1, "data_format string must be longer than 1"
    # Batch
    assert 'N' in data_format, 'Requires at least batch axis: N'
    batch_axis = data_format.index('N')
    # Feature or Channel
    if 'F' in data_format:
        feature_axis = data_format.index('F') - len(data_format)    # Use negative index here!
    elif 'C' in data_format:
        feature_axis = data_format.index('C') - len(data_format)    # Use negative index here!
    else:
        feature_axis = None
    # Sequence
    if 'S' in data_format:
        sequence_axis = data_format.index('S')
    else:
        sequence_axis = None
    # Spatial axes (height, width, x, y, ...)
    spatial_axes = [i for i, a in enumerate(data_format) if a in 'DWHXYZ']
    # Iteration axis
    if 'I' in data_format:
        iter_axis = data_format.index('I')
    else:
        iter_axis = None
    return batch_axis, feature_axis, sequence_axis, spatial_axes, iter_axis


def plot_sample(sample, idx, data_format='NSXYF', geometry=(1.0, 1.0),
                displacement_scale=None, file=None):
    # Extract converged iteration
    if 'I' in data_format:
        sample = sample[..., 0, :]
    # Extract loads and displacements
    displacements = sample[..., idx['Ui']]
    forces = sample[..., idx['Fi']]
    node_id = np.unravel_index(np.argmax(np.abs(displacements)),
                               forces.shape)
    force_dir = node_id[-1]
    u = displacements[:, node_id[1], node_id[2], node_id[3]]
    f = forces[:, node_id[1], node_id[2], node_id[3]]
    mask = np.logical_and(u != 0.0, f != 0.0)
    mask[0] = True
    u = np.concatenate([[0.0], u[mask]])
    f = np.concatenate([[0.0], f[mask]])
    # Calculate scale
    if displacement_scale is None:
        displacement_scale = 0.1 * np.max(geometry) / np.max(
            np.linalg.norm(displacements, axis=1))
    # Plot sample
    num_snapshots = len(u) - 1
    num_columns = 5
    num_rows = int(np.ceil(num_snapshots / num_columns))
    figsize = (7.0, num_rows * 7.0 / num_columns)
    fig = plt.figure(figsize=figsize)
    width_ratios = [1.0] * num_columns + [0.05]
    gs = gridspec.GridSpec(num_rows + 1, num_columns + 1,
                           width_ratios=width_ratios)
    hist_ax = fig.add_subplot(gs[0, 2:3])
    cbar_ax = fig.add_subplot(gs[1:, -1])
    axes = np.asarray(
        [[fig.add_subplot(gs[r + 1, c]) for c in range(num_columns)]
          for r in range(num_rows)])
    # Hysteresis
    plt.sca(hist_ax)
    hist_ax.plot(u, f, 'C1', marker='.')
    hist_ax.spines['left'].set_position('zero')
    hist_ax.spines['bottom'].set_position('zero')
    hist_ax.spines['right'].set_visible(False)
    hist_ax.spines['top'].set_visible(False)
    # xlim = plt.xlim()
    # plt.xlim(0.0, xlim[1] * 1.1)
    # ylim = plt.ylim()
    # plt.ylim(0.0, ylim[1] * 1.1)
    # plt.xlim(1.1 * np.min(displacements), 1.1 * np.max(displacements))
    # plt.ylim(1.1 * np.min(forces), 1.1 * np.max(forces))
    # hist_ax.axis(left=0.4)
    plt.xlabel('$u_{:d}$'.format(force_dir + 1), x=1.0,
               labelpad=0.0)
    plt.ylabel('$F_{:d}$'.format(force_dir + 1), rotation=0, y=1.0,
               labelpad=0.0)
    plt.title('Load-displacement')

    # FE
    geometry = [1.0, 1.0]
    grid_shape = [9, 9]
    E0 = 10000.0
    nu = 0.25
    rho = 0.0
    fy = 15.0
    E1 = 0.0
    thickness = 1.0
    model = MetaModel(result_dir='.', mode='train')
    model.mode = 'generate'
    model.materials.define_elastoplastic(1, E0, nu, rho, fy, E1)
    model.properties.define_2d_section_intelligent_element(1, thickness,
                                                           grid_shape)
    model.generate_beam(geometry, grid_shape, 100, 1, material=1, properties=1)

    # Set intelligent element parameters
    ie = model.elements.instances[1]
    vis = MetaVisualizer(model=model)
    vis.colormap = 'jet'
    vis.climits = [0.0, 15.0]
    vis.static_climits = True
    for i in range(len(u) - 1):
        r, c = np.unravel_index(i, (num_rows, num_columns), order='C')
        plt.sca(axes[r, c])
        # Extract state
        state = ie.decompose_sample(sample[i, ...])
        tpc = ie.draw_element(
            state, stress_type='mises', edgecolor=None,
            displacement_scale=displacement_scale, use_global_coords=False,
            shading='gouraud', draw_connectivity=False, visualizer=vis)
        # Extract load node
        xy = sample[i, node_id[1], node_id[2], idx['Xi']]
        uv_fe = sample[i, node_id[1], node_id[2], idx['Ui']]
        # Draw load node
        xy_fe = xy + displacement_scale * uv_fe
        plt.scatter(xy_fe[:1], xy_fe[1:], color='C1')
        plt.axis('equal')
    for r in range(num_rows):
        for c in range(num_columns):
            if r != num_rows - 1:
                axes[r, c].get_xaxis().set_ticklabels([])
            else:
                axes[r, c].set_xlabel('$X_1$')
            if c != 0:
                axes[r, c].get_yaxis().set_ticklabels([])
            else:
                axes[r, c].set_ylabel('$X_2$', rotation=0, labelpad=12.0)
    # Add colorbar
    cbar = plt.colorbar(mappable=tpc, cax=cbar_ax)
    cbar.solids.set_rasterized(True)
    # Save and close
    if file is not None:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if '.' not in os.path.basename(file):
            file += '.' + matplotlib.rcParams['savefig.format']
            plt.savefig(file)
            os.chmod(file, 0o666)
            plt.close(fig)
    else:
        plt.draw()
        plt.show()


class BaseDict(SortedDict):

    default = None

    def __getitem__(self, key):
        if isinstance(key, (list, tuple)):
            # return all matching keys, raise KeyError if key doesn't exist
            return [self[k] for k in key]
        elif isinstance(key, ndarray):
            return [self[k] for k in key.tolist()]
        elif isinstance(key, slice):
            slice_list = list(range(*key.indices(max(self.keys()) + 1)))
            return [self[k]for k in slice_list]
        else:
            return super(BaseDict, self).__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple, slice)):
            self.update(SortedDict(zip(key, value)))
        elif isinstance(key, ndarray):
            self.update(SortedDict(zip(key.tolist(), value.tolist())))
        else:
            super(BaseDict, self).__setitem__(key, value)

    def __deepcopy__(self, memodict={}):
        items = copy.deepcopy(list(self.items()))
        new = self.__class__(items)
        return new

    @property
    def list(self):
        return self.values()

    @property
    def array(self):
        # TODO: This will fail when nodes have different length
        return np.asarray(self.list)


class Nodes(BaseDict):

    def __init__(self, num_dof_per_node=2, dictionary=None):
        """Defines and manages a node list."""
        if dictionary:
            super(Nodes, self).__init__(dictionary)
        else:
            super(Nodes, self).__init__()
        self.num_dof_per_node = num_dof_per_node

    @property
    def node_coords(self):
        return self.array[:, 1:]

    @property
    def node_dofs(self):
        return np.reshape(
            np.arange(self.num_dof_per_node * len(self), dtype=np.int),
            [-1, self.num_dof_per_node])

    def add_node(self, node_def, node_label=None, tol=1e-8):
        if node_label is None:
            node_label = int(node_def[0])
        if len(self) > 0:
            # Check if node at that coordinate exists
            idx = np.where(
                np.all(np.abs(self.node_coords - node_def[1:]) < tol, axis=1))
            if np.size(idx) == 0:
                # Create new node with desired node_label
                self[node_label] = np.asarray(node_def, dtype=np.float64)
            elif np.size(idx) == 1:
                # Return node_label of existing node
                node_label = int(self.array[idx, 0])
            else:
                raise ValueError(
                    'Multiple nodes at the specified locations found.')
        else:
            # Create new node with desired node_label
            self[node_label] = np.asarray(node_def, dtype=np.float64)
        return node_label

    def find_nodes(self, x=None, y=None, tol=1e-8):
        assert x is not None or y is not None, 'Requires either x or y value.'
        all_nodes = self.array
        if y is not None:
            match = np.isclose(all_nodes[:, 2], y, atol=tol, rtol=0.0)
        else:
            match = [True] * all_nodes.shape[0]
        if x is not None:
            match = np.isclose(all_nodes[match, 1], x, atol=tol, rtol=0.0)
        return np.asarray(all_nodes[match, :])

    def find_node_labels(self, x=None, y=None, tol=1e-8):
        return np.asarray(self.find_nodes(x=x, y=y, tol=tol)[:, 0],
                          dtype=np.int)


class Elements(BaseDict):

    def __init__(self, model, dictionary=None):
        """Defines and manages an element list."""
        if dictionary:
            super(Elements, self).__init__(dictionary)
        else:
            super(Elements, self).__init__()
        self.model = model
        self.instances = BaseDict()

    def __getitem__(self, key):
        if isinstance(key, (list, tuple)):
            # return all matching keys, raise KeyError if key doesn't exist
            return self.__class__(self.nodes,
                                  zip(key, [self[k] for k in key]))
        elif isinstance(key, ndarray):
            return self.__class__(self.nodes,
                                  zip(key, [self[k] for k in key.tolist()]))
        elif isinstance(key, slice):
            slice_list = list(range(*key.indices(max(self.keys()) + 1)))
            return self.__class__(
                self.nodes, zip(slice_list, [self[k] for k in slice_list]))
        else:
            return super(BaseDict, self).__getitem__(key)

    @property
    def nodes(self):
        return self.model.nodes

    @property
    def element_nodes(self):
        return [ed[4:] for ed in self.values()]

    @property
    def element_dofs(self):
        node_dofs = self.nodes.node_dofs
        element_nodes = self.element_nodes
        dofs = [node_dofs[en - 1].ravel() for en in element_nodes]
        return dofs

    def add_element(self, element_def, element_label=None, material_label=None,
                    property_label=None):
        if element_label is None:
            element_label = int(element_def[0])
        if material_label is None:
            material_label = int(element_def[2])
        if property_label is None:
            property_label = int(element_def[3])
        self[element_label] = np.asarray(element_def, dtype=np.int)
        self.instances[element_label] = self.model.instantiate_element(
            element_def)


class Materials(BaseDict):

    def __init__(self, dictionary=None):
        """Defines and manages a material dictionary."""
        if dictionary:
            super(Materials, self).__init__(dictionary)
        else:
            super(Materials, self).__init__()

    def define_elastic(self, material_label, E, nu, rho):
        """Define linear elastic material parameters."""
        mat_type = 1
        self[material_label] = [mat_type, E, nu, rho]

    def define_elastoplastic(self, material_label, E, nu, rho, fy, Et):
        """Define linear elasto-plastic material parameters."""
        mat_type = 11
        self[material_label] = [mat_type, E, nu, rho, fy, Et]

    def define_linear_visco_elastic(self, material_label, nu, par1, par2, par3):
        """Define linear visco-elastic material parameters."""
        mat_type = 21
        self[material_label] = [mat_type, nu, par1, par2, par3]

    def define_hyper_elastic_neo_hooke(self, material_label, nu, par1, par2,
                                       par3):
        """Define neo hooke material parameters."""
        mat_type = 31
        self[material_label] = [mat_type, nu, par1, par2, par3]


class Properties(BaseDict):

    def __init__(self, dictionary=None):
        """Defines and manages a material dictionary."""
        if dictionary:
            super(Properties, self).__init__(dictionary)
        else:
            super(Properties, self).__init__()

    def define_2d_section(self, property_label, thickness):
        """Define properties for thickness of a 2D element."""
        prop_type = 1
        self[property_label] = [prop_type, thickness]

    def define_2d_section_intelligent_element(self, property_label, thickness,
                                              grid_shape, element_dict=None):
        """Define properties for 2D intelligent elements."""
        prop_type = 101
        if element_dict is None:
            element_dict = {}
        self[property_label] = [prop_type, thickness, grid_shape, element_dict]


class BC(object):

    def __init__(self, nodes):
        self.dirichlet = Dirichlet()
        self.neumann = Neumann()
        self._nodes = nodes

    @property
    def num_lock_dofs(self):
        return np.sum(self.dirichlet.locks.array[:, 1:], dtype=np.int)

    @property
    def lock_dofs(self):
        lock_array = self.dirichlet.locks.array
        lock_node_labels = lock_array[:, 0]
        lock_node_dofs = self._nodes.node_dofs[lock_node_labels - 1, :].ravel()
        lock_dofs = np.asarray(lock_array[:, 1:].ravel(), dtype=np.bool)
        return lock_node_dofs[lock_dofs]

    @property
    def reduce_dofs(self):
        node_dofs = self._nodes.node_dofs.ravel().tolist()
        lock_dofs = self.lock_dofs
        reduce_dofs = [nd for nd in node_dofs if nd not in lock_dofs]
        return np.asarray(reduce_dofs)


class Dirichlet(BaseDict):

    def __init__(self, *args, **kwargs):
        super(Dirichlet, self).__init__(*args, **kwargs)
        self.locks = BaseDict()

    def set(self, node_label, locks, values):
        self.locks[node_label] = np.asarray([node_label] + locks, dtype=np.int)
        self[node_label] = np.asarray([node_label] + values, dtype=np.float64)


class Neumann(BaseDict):

    def set(self, node_label, values):
        self[node_label] = np.asarray([node_label] + values, dtype=np.float64)


class Physics(object):

    @staticmethod
    def extract_stress_values(stresses, stress_type='mises'):
        """Calculate scalar stress values along the last axis of stresses."""
        if stress_type == 'mises':
            if stresses.shape[-1] == 6:
                sig_xx = stresses[..., 0]
                sig_yy = stresses[..., 1]
                sig_zz = stresses[..., 2]
                tau_xy = stresses[..., 3]
                tau_yz = stresses[..., 4]
                tau_zx = stresses[..., 5]
            elif stresses.shape[-1] == 3:
                # Plane stress case
                sig_xx = stresses[..., 0]
                sig_yy = stresses[..., 1]
                sig_zz = 0.0
                tau_xy = stresses[..., 2]
                tau_yz = 0.0
                tau_zx = 0.0
            else:
                raise NotImplementedError
            sig_scalar = np.sqrt(
                0.5 * ((sig_xx - sig_yy) ** 2
                       + (sig_yy - sig_zz) ** 2
                       + (sig_zz - sig_xx) ** 2
                       + 6 * (tau_xy ** 2 + tau_yz ** 2 + tau_zx ** 2)))
        elif isinstance(stress_type, int):
            sig_scalar = stresses[..., stress_type]
        else:
            raise ValueError('Invalid stress type: {:s}'.format(
                str(stress_type)))
        return sig_scalar

    @staticmethod
    def extract_rigid_body_motion(xi, ui):
        mask = ui == None
        ui = np.ma.array(ui, mask=mask, dtype=np.float64)
        # Rigid body translation
        urb = np.mean(ui, axis=0)
        # Rigid body rotation
        mu_xi = np.mean(xi, axis=0)
        xc = xi - mu_xi
        xc_ui = np.copy(xc)
        xc_ui[~mask] += np.asarray(ui[~mask], dtype=np.float64)
        xc_ui -= urb
        A_ui = np.cross(xc, xc_ui, axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mask = np.abs(A_ui) > 10.0 * np.finfo('float64').eps  # mask zeros
        if np.any(mask):
            A_ui = A_ui[mask]
            phi = np.arcsin(A_ui / np.linalg.norm(xc[mask, :], axis=1)
                            / np.linalg.norm(xc_ui[mask, :], axis=1))
            n = A_ui / np.linalg.norm(A_ui)
            phirb = np.mean(phi, axis=0)
            nrb = np.mean(n, axis=0)
            Rrb = np.asarray([[np.cos(phirb), -np.sin(phirb)],
                              [np.sin(phirb), np.cos(phirb)]])
        else:
            Rrb = np.identity(xc.shape[-1])
        return urb, Rrb

    @staticmethod
    def remove_rigid_body_motion(xi, ui, urb, Rrb, rotate_displacements=True,
                                 rotate_translation=False):
        """Add rigid body translation and rotation to fe results."""
        un = np.copy(ui)
        un = np.ma.array(un, dtype=np.float64, mask=(un == None))
        # Remove rigid body translation
        if rotate_translation:
            un -= Physics.rotate_displacements(urb, Rrb)
        else:
            un -= urb
        # Remove rigid body rotation
        mu_xi = np.mean(xi, axis=0)
        xc = xi - mu_xi
        un -= np.dot(xc, Rrb.T) - xc
        # Rotate displacements to new orientation
        if rotate_displacements:
            un = np.ma.array(Physics.rotate_displacements(
                un.filled(0.0), Rrb.T), mask=un.mask)
        # Replace Nans with nones
        mask = un.mask
        if np.any(mask):
            un = np.asarray(un, dtype=object)
            un[mask] = None
        else:
            un = np.asarray(un, dtype=np.float64)
        return un

    @staticmethod
    def add_rigid_body_motion(xi, un, urb, Rrb, rotate_displacements=True,
                              rotate_translation=False):
        """Add rigid body translation and rotation to fe results."""
        ui = np.copy(un)
        ui = np.asarray(ui, dtype=np.float64)   # Turn None into NaN
        # Rotate displacements to new orientation
        if rotate_displacements:
            ui = Physics.rotate_displacements(ui, Rrb)
        # Add rigid body translation
        if rotate_translation:
            ui += Physics.rotate_displacements(urb, Rrb)
        else:
            ui += urb
        # Add rigid body rotation
        mu_xi = np.mean(xi, axis=0)
        xc = xi - mu_xi
        ui += np.dot(xc, Rrb.T) - xc
        # Replace Nans with nones
        mask = np.isnan(ui)
        if np.any(mask):
            ui = np.asarray(ui, dtype=object)
            ui[mask] = None
        else:
            ui = np.asarray(ui, dtype=np.float64)
        return ui

    @staticmethod
    def rotate_displacements(ui, R):
        return np.dot(ui, R.T)

    @staticmethod
    def rotate_forces(fi, R):
        return np.dot(fi, R.T)

    @staticmethod
    def rotate_stresses(si, R):
        if R.shape[-1] == 3:
            raise NotImplementedError('Rotation in 3D not implemented!')
        if si.shape[-1] == 3:
            # Ordering: S11, S22, S12
            i12 = 2
        elif si.shape[-1] == 6:
            # Ordering: S11, S22, S33, S12, S23, S13
            i12 = 3
        R = np.identity(si.shape[-1])
        R[0, 0] = R[0, 0] ** 2
        R[1, 0] = R[1, 0] ** 2
        R[2, 0] = - R[1, 0] * R[0, 0]
        R[0, 1] = R[0, 1] ** 2
        R[1, 1] = R[1, 1] ** 2
        R[2, 1] = - R[0, 1] * R[1, 1]
        R[0, 2] = - 2.0 * R[0, 1] * R[0, 0]
        R[1, 2] = - 2.0 * R[1, 0] * R[0, 0]
        R[2, 2] = R[1, 1] ** 2 - R[0, 1] ** 2
        return np.dot(si, R.T)

    @staticmethod
    def safe_divide(a, b, tol=None):
        if tol is None:
            tol = 10.0 * max(np.finfo(a.dtype).eps, np.finfo(b.dtype).eps)
        return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > tol)


class BaseElement(Physics):

    element_type = 0

    def __init__(self, fe_model, element_definition, **kwargs):
        """An abstract parent class for finite element.

        Args:
            fe_model:               The FE model the element belongs to
            element_definition:     An iterable of numbers defining element
                                    label, type, material, properties and nodes

        """
        warnings.simplefilter('always', UserWarning)
        warnings.filterwarnings('ignore', message='findfont: Font family')
        assert isinstance(element_definition[1], (str, int, np.int64, np.int32))
        assert str(self.element_type) in str(element_definition[1]), \
            'Invalid element definition type: {:d} != {:d}'.format(
                element_definition[1], self.element_type)
        self._fe_model = fe_model
        self.label = int(element_definition[0])
        self.material_label = int(element_definition[2])
        self.property_label = int(element_definition[3])
        self.node_labels = list(element_definition[4:])
        self.num_grid_dims = 2
        self.num_generalized_dims = 0
        self.num_stress_components = 3
        self.ref_node_labels = self.node_labels[:4]     # TODO: what to do with unused points?

    def __eq__(self, other):
        """Compare element type with other element or integer."""
        try:
            element_type = self.element_type  # Must be set in __init__
        except AttributeError:
            raise AttributeError(
                'self.element_type must be set in element class __init__.')
        if isinstance(other, self.__class__):
            eq = True
        elif isinstance(other, (int, float)):
            eq = other.__eq__(element_type)
        else:
            raise ValueError(
                'Equality of elements with instances of class {:s} not defined.'
                    .format(self.__class__.__name__))
        return eq

    @property
    def material(self):
        """Get material parameters from global dictionary."""
        assert self.material_label in self._fe_model.materials.keys(), \
            'Element material {:d} not defined.'.format(self.material_label)
        return self._fe_model.materials[self.material_label]

    @property
    def properties(self):
        """Get material parameters from global dictionary."""
        assert self.property_label in self._fe_model.properties.keys(), \
            'Element material {:d} not defined.'.format(self.property_label)
        return self._fe_model.properties[self.property_label]

    @property
    def nodes(self):
        """Get node coords from global dictionary."""
        for n in self.node_labels:
            assert n in self._fe_model.nodes.keys(), \
                'Element node {:d} not defined.'.format(n)
        return self._fe_model.nodes[self.node_labels]

    @property
    def num_nodes(self):
        return len(self.node_labels)

    @property
    def num_dims(self):
        return self.num_grid_dims + self.num_generalized_dims

    @property
    def num_dofs(self):
        return self.num_nodes * self.num_dims

    @property
    def rotation_matrix(self):
        """Calculate the rotation matrix from local to global coordinates."""
        num_spatial_dims = self.num_grid_dims
        num_generalized_dims = self.num_generalized_dims
        num_nodes = self.num_nodes
        element_nodes = np.stack(self.nodes, axis=0)
        ref_node_labels = self.ref_node_labels
        # Extract reference element node coordinates
        ref_element_nodes = np.concatenate(
            [element_nodes[element_nodes[:, 0] == l, :]
             for l in ref_node_labels],
            axis=0)
        ref_element_coords = ref_element_nodes[:, 1:]
        # Element local axis 1 defined by connection of first to second node
        local_x1 = np.diff(ref_element_coords[[0, 1], :num_spatial_dims],
                           axis=0)
        local_x1 = np.squeeze(local_x1)
        local_x1 /= np.linalg.norm(local_x1)
        # Element plane is defined by connection of first to last node
        tmp_x2 = np.diff(ref_element_coords[[0, -1], :num_spatial_dims],
                         axis=0)
        tmp_x2 = np.squeeze(tmp_x2)
        # Append third dimension
        if num_spatial_dims == 2:
            local_x1 = np.concatenate([local_x1, [0.0]])
            tmp_x2 = np.concatenate([tmp_x2, [0.0]])
        # Ensure right hand cartesian coordinate system
        local_x3 = np.cross(local_x1, tmp_x2)
        local_x3 /= np.linalg.norm(local_x3)
        local_x2 = np.cross(local_x3, local_x1)
        local_x2 /= np.linalg.norm(local_x2)
        # Assemble rotation matrix
        local_R_base = np.stack([local_x1, local_x2, local_x3], axis=1)
        # Remove unnecessary dimensions
        if num_spatial_dims == 2:
            local_R_base = local_R_base[:num_spatial_dims, :num_spatial_dims]
        # Invert (orthogonal, so transpose works)
        R_base = local_R_base.T
        # Physical space rotations are identity transform for generalized coords
        R_generalized = np.identity(num_spatial_dims + num_generalized_dims)
        R_generalized[:num_spatial_dims, :num_spatial_dims] = R_base
        # Place block diagonals along axis (matrix kronecker product)
        R = np.kron(np.identity(num_nodes), R_generalized)
        return R

    def loads(self, ue=None):
        """Calculate the loads of the element in global coordinates."""
        # Transform displacements to local coordinates
        R = self.rotation_matrix
        if ue is not None:
            ue_local = np.reshape(np.dot(R.T, np.ravel(ue, order='C')),
                                  [-1, self.num_dims], order='C')
        else:
            ue_local = None
        # Calculate local loads
        fe_local = self.local_loads(ue_local=ue_local)
        # Transform
        fe = np.reshape(np.dot(R, np.ravel(fe_local, order='C')),
                        [-1, self.num_dims], order='C')
        return fe

    def stiffness(self, ue=None):
        """Calculate the stiffness of the element in global coordinates."""
        # Transform displacements to local coordinates
        R = self.rotation_matrix
        if ue is not None:
            ue_local = np.reshape(np.dot(R.T, np.ravel(ue, order='C')),
                                  [-1, self.num_dims], order='C')
        else:
            ue_local = None
        # Calculate local stiffness
        Ke_local = self.local_stiffness(ue_local=ue_local)
        # Transform
        Ke = np.dot(np.dot(R.T, Ke_local), R)
        return Ke

    def mass(self):
        """Calculate the stiffness of the element in global coordinates."""
        # Transform displacements to local coordinates
        R = self.rotation_matrix
        # Calculate local stiffness
        Me_local = self.local_mass()
        # Transform
        Me = np.dot(np.dot(R.T, Me_local), R)
        return Me

    def local_loads(self, ue_local=None):
        """Calculate the loads of the element in local coordinates."""
        if ue_local is not None:
            warnings.warn('Not implemented!')
            return np.ones_like(ue_local)
        else:
            raise NotImplementedError

    def local_stiffness(self, ue_local=None):
        """Calculate the stiffness of the element in local coordinates."""
        if ue_local is not None:
            warnings.warn('Not implemented!')
            return np.identity(ue_local.size)
        else:
            raise NotImplementedError

    def local_mass(self):
        raise NotImplementedError


class DummyElement(BaseElement):

    DEBUG = False
    DEBUG_PLOT = False
    DEBUG_DISABLE_COROTATIONAL = False
    DEBUG_COMPARE = False
    VERBOSITY = 2
    INDENT = ''

    element_type = 100
    base_name = 'die'

    def __init__(self, fe_model, element_definition, mode=None):
        """An linear intelligent convolutional continuum meta element.

        Args:
            fe_model:               Owner finite element model of the element
            element_definition:     Element definition array

        """
        super(DummyElement, self).__init__(
            fe_model, element_definition)

        ############################
        # Internal grid
        #

        grid_shape = self.properties[2]
        self.grid_shape = list(grid_shape)
        self.grid_size = np.prod(self.grid_shape)
        self.num_grid_dims = len(grid_shape)
        self.num_stress_components = (self.num_grid_dims
                                      * (self.num_grid_dims + 1) // 2)
        assert self.num_nodes == (self.grid_size
                                  - np.prod(np.asarray(self.grid_shape) - 2))
        # Ensure grid_labels first axis corresponds with nodes
        self.grid_labels = calc_grid_labels(grid_shape)
        self.boundary_grid_labels = extract_boundary_grid_labels(
            self.grid_labels, grid_shape)
        # find node labels of corner and edges of the internal grid
        corner_grid_labels = self.grid_labels[tuple(slice(None, None, j - 1)
                                                    for j in self.grid_shape)]
        self.ref_grid_labels = np.ravel(corner_grid_labels, order='F')
        self.ref_node_labels = self.grid_to_node(self.ref_grid_labels)

        # The filled area/volume of the grid is defined as a "binary" ndarray
        self.body = np.ones(self.grid_shape)   # floats allow a 'density'
        self._init_params_fe()
        self._init_params_nn()

    def _init_params_fe(self):
        """Set all object parameters for the internal finite element model."""

        # Geometry
        coords = np.asarray(self.nodes)[:, 1:]
        self.geometry = np.abs(np.max(coords, axis=0) - np.min(coords, axis=0))

    def _init_params_nn(self):
        """Set all object parameters for the internal neural network."""
        # Data parameters
        if self.num_grid_dims == 2:
            self.data_format = 'NSXYF'
            self.param_features = ['E', 'lx', 'ly']
        elif self.num_grid_dims == 3:
            self.data_format = 'NSXYZF'
            self.param_features = ['E', 'lx', 'ly', 'lz']
        else:
            raise ValueError('Need spatial dimension of 2 or 3.')
        assert self.param_features == sorted(self.param_features), \
            'param_features must be given in sorted order.'
        # Define feature order
        self.feature_components = []
        self.feature_components += ['Xi_boundary'] * self.num_grid_dims
        self.feature_components += ['Ui_boundary'] * self.num_grid_dims
        self.feature_components += ['boundary']
        self.feature_components += ['body']
        self.feature_components += ['Xi'] * self.num_grid_dims
        self.feature_components += ['Ui'] * self.num_grid_dims
        self.feature_components += ['Si'] * self.num_stress_components
        self.feature_components += ['Fi'] * self.num_grid_dims
        self.feature_components += self.param_features                          # Must be last and sorted!!!
        # Define input and output features
        self.features = list(OrderedDict.fromkeys(self.feature_components))     # THIS KEEPS ORDER!!!!!1111
        self.input_features = ['Ui_boundary']
        self.output_features = ['Ui', 'Si']
        self.output_features += ['Fi']                  # Keep this last
        # Convert feature order to indices
        self.idx = dict()
        # Cycle through each unique feature
        for f in set(self.feature_components):
            feature_indices = [i for i, ff in enumerate(self.feature_components)
                               if f == ff]  # Gather all associated indices
            self.idx[f] = feature_indices
        self.input_idx = []
        for f in self.input_features:
            self.input_idx.extend(self.idx[f])
        self.output_idx = []
        for f in self.output_features:
            self.output_idx.extend(self.idx[f])
        if self.DEBUG:
            print('features', self.feature_components)
            print('input_features', self.input_features)
            print('output_features', self.output_features)
            print('idx', self.idx)
            print('input_idx', self.input_idx)
            print('output_idx', self.output_idx)
        # Get input and output feature components
        self.input_feature_components = [
            f
            for f in self.input_features
            for _ in range(self.feature_components.count(f))]   # TODO: slow
        self.output_feature_components = [
            f
            for f in self.output_features
            for _ in range(self.feature_components.count(f))]   # TODO: slow
        # Calculate indices of field variables in the input tensor
        self.ix = dict()
        for fi in self.input_features:
            self.ix[fi] = [
                i for i, f in enumerate(self.input_feature_components)
                if f == fi]
        # Calculate indices of field variables in the target/prediction tensor
        self.iy = dict()
        for fo in self.output_features:
            self.iy[fo] = [
                i for i, f in enumerate(self.output_feature_components)
                if f == fo]
        # Extract data features
        self.num_features = len(self.feature_components)
        self.data_shape = [None] + list(self.grid_shape) + [self.num_features]

    @property
    def origin_coords(self):
        nodes = np.asarray(self.nodes)
        idx = self.node_labels.index(self.ref_node_labels[0])
        return nodes[idx, 1:]

    def grid_to_node(self, grid):
        """Convert labels from the internal grid to the element node labels."""
        boundary_grid_labels = np.asarray(self.boundary_grid_labels)
        node_labels = np.asarray(self.node_labels)
        node = [np.sum(np.where(boundary_grid_labels == g, node_labels, 0))
                for g in np.ravel(grid, order='F')]
        return np.asarray(node)

    def node_to_grid(self, node):
        """Convert element node labels to labels of the internal grid."""
        boundary_grid_labels = np.asarray(self.boundary_grid_labels)
        node_labels = np.asarray(self.node_labels)
        grid = [np.sum(np.where(node_labels == n, boundary_grid_labels, 0))
                for n in np.ravel(node, order='F')]
        return np.asarray(grid)

    @property
    def grid(self):
        return generate_grid(self.geometry, self.grid_shape)

    @property
    def grid_global_coords(self):
        grid_global_coords = np.copy(self.grid)
        # Add rotation
        R = self.rotation_matrix[:self.num_dims, :self.num_dims]
        for i, x in enumerate(grid_global_coords):
            grid_global_coords[i, :] = np.dot(R, x)
        # Add origin
        grid_global_coords = grid_global_coords + self.origin_coords
        return grid_global_coords

    def assemble_sample(self, fe_results, features=None, gridencode=True):
        """Assemble the finite element response into a sample tensor."""
        if features is None:
            features = self.features
        # Filter out non-required state variables
        point_dict = {k: v for k, v in fe_results.items()
                      if isinstance(v, (np.ndarray, int, float))}
        # Assemble into grid
        if gridencode:
            # Ensure coordinates are present    # TODO: merge with else
            if 'Xi' not in point_dict.keys():
                point_dict['Xi'] = self.grid
            # Grid encoding (voxel/pixel)
            _, body, grid_dict = grid_encode(
                self.grid, self.grid_shape, point_dict['Xi'], point_dict)     # TODO: Does this leave the origin as outside of body when no loading?
            body, sample_dict = repair_voids(body, grid_dict)
            boundary = find_boundary(body)
            X_boundary = sample_dict['Xi'] * boundary
            U_boundary = sample_dict['Ui'] * boundary
        else:
            # Ensure coordinates are present    # TODO: merge with else
            if 'Xi' not in point_dict.keys():
                point_dict['Xi'] = self.grid
            if 'Xe_local' not in point_dict.keys():
                point_dict['Xe_local'] = self.grid[
                                         self.boundary_grid_labels - 1, :]
            # Reshape internal grid variables to grid size
            sample_dict = {f: np.reshape(
                point_dict[f], self.grid_shape + [-1],
                order='F')
                for f in features
                if f not in (['boundary', 'body', 'Xi_boundary', 'Ui_boundary']
                             + self.param_features)}
            # Assemble meta element nodes into internal boundary
            gi = np.asarray(self.boundary_grid_labels) - 1
            gii = np.unravel_index(gi, self.grid_shape, order='F')    # Must be Fortran order!!!
            gii = tuple(np.asarray(g, dtype=np.int32) for g in gii)   # use int32 for C++
            X_boundary = np.zeros(self.grid_shape + [self.num_grid_dims])
            U_boundary = np.zeros(self.grid_shape + [self.num_grid_dims])
            boundary = np.zeros(self.grid_shape + [1])
            X_boundary[gii] = point_dict['Xe_local']
            U_boundary[gii] = point_dict['Ue_local']
            boundary[gii] = 1.0
            body = self.body[..., np.newaxis]
        # Calculate boundary information
        sample_dict['Xi_boundary'] = X_boundary
        sample_dict['Ui_boundary'] = U_boundary
        sample_dict['body'] = body
        sample_dict['boundary'] = boundary
        # Append params
        geometry = list(self.geometry)
        material = list(self.material[1:])
        sample_dict['lx'] = geometry[0]
        sample_dict['ly'] = geometry[1]
        if self.num_grid_dims > 2:
            sample_dict['lz'] = geometry[2]
        sample_dict['E'] = material[0]
        # Assemble sample tensor
        sample_shape = [d for d, a in zip(self.data_shape, self.data_format)
                        if a not in 'NSI']
        sample = np.zeros(sample_shape)
        for f in features:
            sample[..., self.idx[f]] = sample_dict[f]
        return sample

    def decompose_output(self, y):
        """Decompose the neural network output to pointwise state."""
        return self._decompose_tensor(y, self.iy, self.output_features)

    def decompose_sample(self, sample):
        """Decompose the neural network sample to pointwise state."""
        return self._decompose_tensor(sample, self.idx, self.features)

    def _decompose_tensor(self, sample, idx, features):
        # Flatten all axes but the feature axis
        feature_shape = [-1]
        decomposed = {k: (np.reshape(sample[..., idx[k]],
                                     feature_shape + [len(idx[k])],
                                     order='F')
                          if k not in self.param_features
                          else np.ravel(sample[..., idx[k]])[0])
                      for k in features}
        # Ensure coordinates are given
        if 'Xi' not in decomposed.keys():
            decomposed['Xi'] = self.grid
        # Extract element nodal data
        #    Only in output features since they are complete field variables
        gi = np.asarray(self.boundary_grid_labels) - 1
        decomposed.update(
            {'{:s}_local'.format(k.replace('i', 'e')): decomposed[k][gi, ...]
             for k in self.output_features})
        return decomposed

    def plot_sample(self, sample, file, displacement_scale=None):
        file = copy.deepcopy(file)
        if displacement_scale is None:
            displacement_scale = (0.1 * np.max(self.geometry)
                                  / (3.0 * self.bc_value_std))
        # Plot all states if sample is a history
        if ('S' in self.data_format
                and len(sample.shape) == len(self.data_format) - 1):
            for s in range(sample.shape[0]):
                if '.' in os.path.basename(file):
                    tmp_file = os.path.join(
                        os.path.dirname(file),
                        os.path.basename(file)
                            .replace('.', '_{:03d}.'.format(s)))
                else:
                    tmp_file = file + '_{:03d}'.format(s)
                self.plot_sample(sample[s, ...], tmp_file,
                                 displacement_scale=displacement_scale)
            return
        state = self.decompose_sample(sample)
        vis = MetaVisualizer(model=self._fe_model)
        vis.climits = [0.0, fy]
        vis.static_climits = True
        if 'Ui0' in state.keys():
            state0 = dict()
            state0['Xi'] = self.grid
            state0['Ui'] = state['Ui0']
            state0['Fi'] = state['Fi0']
            state0['Si'] = state['Si0']
            if 'Pi0' in state.keys():
                state0['Pi'] = state['Pi0']
            state0['Ui0'] = state['Ui0']
            state0['Fi0'] = state['Fi0']
            state0['Si0'] = state['Si0']
            if 'Pi0' in state.keys():
                state0['Pi0'] = state['Pi0']
            state0 = self.decompose_sample(self.assemble_sample(state0))
            if self.material[0] == 11:
                vmax = self.material[4]
                vmin = 0.0
            else:
                stress_values = [
                    self.extract_stress_values(f['Si'], stress_type='mises')
                    if (f is not None and f['Si'] is not None)
                    else 0.0
                    for f in [state, state0]]
                vmax = max(np.max(stress_values), 0.0)
                vmin = min(np.min(stress_values), 0.0)
            vis.climits = [vmin, vmax]
            vis.element_alpha = 1.0
            fig = plt.figure(figsize=(7.0, 4.0))
            axes = AxesGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.35,
                            share_all=True, aspect=True, cbar_mode='single',
                            cbar_location='top', cbar_pad=0.2, cbar_size='3%')
            plt.sca(axes[0])
            tpc0 = self.draw_element(state=state0, stress_type='mises',
                                     visualizer=vis, edgecolor='k',
                                     shading='gouraud', draw_loads=True,
                                     draw_displacements=True,
                                     displacement_scale=displacement_scale)
            vis.element_alpha = 1.0
        else:
            fig = plt.figure(figsize=(4.0, 4.0))
            axes = AxesGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.35,
                            share_all=True, aspect=True, cbar_mode='each',
                            cbar_location='top', cbar_pad=0.2, cbar_size='3%')
        plt.sca(axes[-1])
        # TODO
        if self.material[0] == 11:
            vmax = self.material[4]
            vmin = 0.0
            vis.climits = [vmin, vmax]
        tpc = self.draw_element(state=state, stress_type='mises',
                                edgecolor='k', visualizer=vis,
                                shading='gouraud', draw_loads=True,
                                draw_displacements=True,
                                displacement_scale=displacement_scale)
        cbar = axes[-1].cax.colorbar(mappable=tpc)
        cbar.solids.set_rasterized(True)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # plt.tight_layout()
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9)
        min_coords = np.min(self.grid_global_coords, axis=0)
        max_coords = np.max(self.grid_global_coords, axis=0)
        delta = 0.1 * np.max(self.geometry)
        plt.axis([min_coords[0] - delta, max_coords[0] + delta,
                  min_coords[1] - delta, max_coords[1] + delta])
        if file is not None:
            os.makedirs(os.path.dirname(file), exist_ok=True)
            if '.' not in os.path.basename(file):
                file += '.' + matplotlib.rcParams['savefig.format']
            plt.savefig(file)
            os.chmod(file, 0o666)
            plt.close(fig)

    def plot_sample_features(self, sample, file, displacement_scale=None,
                             force_scale=None, disable_cbar=False):
        base_file = copy.deepcopy(file)
        state = self.decompose_sample(sample)
        vis = MetaVisualizer(model=self._fe_model)
        if displacement_scale is None:
            base_displacement_scale = (0.08 * np.max(self.geometry)
                                       / np.max(state['Ui']))
        else:
            base_displacement_scale = displacement_scale
        if force_scale is None:
            base_force_scale = (0.09 * np.max(self.geometry)
                                           / np.max(state['Fi']))
        else:
            base_force_scale = force_scale
        base_cmap = vis.colormap
        if 'Ui0' in state.keys():
            state0 = dict()
            state0['Xi'] = self.grid
            state0['Ui'] = state['Ui0']
            state0['Fi'] = state['Fi0']
            state0['Si'] = state['Si0']
            if 'Pi0' in state.keys():
                state0['Pi'] = state['Pi0']
            state0['Ui0'] = state['Ui0']
            state0['Fi0'] = state['Fi0']
            state0['Si0'] = state['Si0']
            if 'Pi0' in state.keys():
                state0['Pi0'] = state['Pi0']
            state0 = self.decompose_sample(self.assemble_sample(state0))
            stress_values = [
                self.extract_stress_values(f['Si'], stress_type='mises')
                if (f is not None and f['Si'] is not None)
                else 0.0
                for f in [state, state0]]
            vmax = max(np.max(stress_values), 0.0)
            vmin = min(np.min(stress_values), 0.0)
            vis.climits = [vmin, vmax]
        min_coords = np.min(self.grid_global_coords, axis=0)
        max_coords = np.max(self.grid_global_coords, axis=0)
        delta = 0.1 * np.max(self.geometry)
        for feature_name, feature_array in state.items():
            if feature_name in self.param_features:
                continue
            if '_local' in feature_name:
                continue
            draw_loads = 'F' in feature_name
            if '0' in feature_name:
                cmap = cm.get_cmap(base_cmap, 512)
                new_values = (np.asarray([1.5, 1.5, 1.5, 1.0])
                              * cmap(np.linspace(0.0, 1.0, 512)))
                new_values[new_values > 1.0] = 1.0
                vis.colormap = ListedColormap(new_values)
            else:
                vis.colormap = base_cmap
            if 'U' in feature_name:
                displacement_scale = base_displacement_scale
            else:
                displacement_scale = 0.0
            draw_displacements = 'U' in feature_name
            if feature_name == 'body':
                facecolor = 'C2'
            elif '_boundary' in feature_name:
                facecolor = 'w'
            else:
                facecolor = 'gainsboro'
            if feature_name == 'boundary':
                edgecolor = 'C2'
            else:
                edgecolor = 'k'

            if 'S' in feature_name or 'P' in feature_name:
                shading = 'gouraud'
            else:
                shading = 'uniform'
            if np.shape(feature_array)[-1] > 2:
                num_plots = np.shape(feature_array)[-1]
            else:
                num_plots = 1
            for ip in range(num_plots):
                if num_plots > 1:
                    feature_index_name = feature_name + '-{:d}'.format(ip)
                else:
                    feature_index_name = feature_name
                fig = plt.figure(figsize=(4.0, 4.0))
                if shading == 'uniform' or disable_cbar:
                    axes = AxesGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.35,
                                    share_all=True, aspect=True)
                else:
                    axes = AxesGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.35,
                                    share_all=True, aspect=True,
                                    cbar_mode='each',
                                    cbar_location='top', cbar_pad=0.2,
                                    cbar_size='3%')
                plt.sca(axes[0])
                tpc = self.draw_element(state=state, stress_type=ip,
                                        edgecolor=edgecolor,
                                        facecolor=facecolor,
                                        visualizer=vis, shading=shading,
                                        draw_loads=draw_loads,
                                        draw_displacements=draw_displacements,
                                        displacement_scale=displacement_scale)
                if tpc is not None:
                    cbar = axes[-1].cax.colorbar(mappable=tpc)
                    cbar.solids.set_rasterized(True)
                plt.subplots_adjust(left=0.075, right=0.95, top=0.9)
                plt.axis([min_coords[0] - delta, max_coords[0] + delta,
                          min_coords[1] - delta, max_coords[1] + delta])
                file = copy.deepcopy(base_file)
                os.makedirs(os.path.dirname(file), exist_ok=True)
                if '.' not in os.path.basename(file):
                    file += '_{:s}'.format(feature_index_name)
                    file += '.' + matplotlib.rcParams['savefig.format']
                else:
                    file += os.path.join(
                        os.path.dirname(file), os.path.basename(file).replace(
                            '.', '_{:s}.'.format(feature_index_name)))
                plt.savefig(file)
                os.chmod(file, 0o666)
                plt.close(fig)

    def draw_element(self, state=None, visualizer=None, stress_type=None,
                     displacement_scale=None,  use_global_coords=False,
                     edgecolor=None, facecolor=None, shading='gouraud',
                     draw_connectivity=False, draw_loads=False,
                     draw_displacements=False):
        """Draws the element into the current axes."""
        state = state or self.fe_state
        visualizer = visualizer or MetaVisualizer(self._fe_model)
        displacement_scale = displacement_scale or visualizer.displacement_scale
        edgecolor = edgecolor or visualizer.element_edgecolor
        facecolor = facecolor or visualizer.element_color
        # Extract data
        Xe_local = np.copy(np.asarray(self.nodes)[:, 1:])
        if not use_global_coords:
            Xe_local -= self.origin_coords
        if state and stress_type is not None:
            if use_global_coords:
                Xi = np.copy(self.grid_global_coords)
            elif np.all(state['Xi'] == 0.0):
                Xi = np.copy(self.grid)
            else:
                Xi = np.reshape(np.copy(state['Xi']),
                                [np.prod(self.grid_shape), -1], order='F')
            Ui = np.reshape(np.copy(state['Ui']),
                            [np.prod(self.grid_shape), -1], order='F')
            Si = np.reshape(np.copy(state['Si']),
                            [np.prod(self.grid_shape), -1], order='F')
            Ue_local = np.copy(state['Ue_local'])
        else:
            Ue_local = 0.0
        # TODO: rotations need to be applied to X, U, S
        # Add outer deformation
        Xe_local += displacement_scale * Ue_local
        # Sort outer coords and displacements mathematically positive
        Xe_local = np.copy(sort_nodes_positive_rotation(
            Xe_local, self.grid_shape))
        outer_path = visualizer.assemble_path(Xe_local)
        # Draw
        if not shading == 'uniform' and state and stress_type is not None:
            # Add inner deformation
            Xi += displacement_scale * Ui
            # Get inner coords and values
            inner_S_component = self.extract_stress_values(
                Si, stress_type=stress_type)
            # Draw gradient patch
            tpc = visualizer.draw_gradient_patch(
                outer_path, Xi, inner_S_component,
                edgecolor=edgecolor, shading=shading)
        else:
            # Draw uniformly colored patch
            visualizer.draw_uniform_patch(
                outer_path, facecolor=facecolor, edgecolor=edgecolor)
            tpc = None
        # Plot internal element grid
        if draw_connectivity and self.fenics_interface:
            if (self.fenics_interface.vertices is not None
                    and self.fenics_interface.connectivity is not None):
                vertices = self.fenics_interface.vertices
                connectivity = self.fenics_interface.connectivity
                # Vertices are in a different order than the mesh coordinates
                #   (midpoint nodes missing, order defined by function space)
                order = np.concatenate(
                    [np.where(np.all(Xi == v, axis=1))[0] for v in vertices])
                # Add deformations
                vertices_deformed = vertices + displacement_scale * Ui[order, :]
                visualizer.draw_internal_grid(vertices_deformed, connectivity)
        # Plot loads
        if draw_loads:
            Fi = np.copy(state['Fi'])
            visualizer.draw_force_field(Xi[:, 0], Xi[:, 1], Fi[:, 0], Fi[:, 1])
        # Plot displacements
        if draw_displacements:
            visualizer.draw_displacement_field(Xi[:, 0], Xi[:, 1],
                                               displacement_scale * Ui[:, 0],
                                               displacement_scale * Ui[:, 1])
        return tpc


class MetaModel(object):

    DEBUG = False
    INDENT = ''

    def __init__(self, result_dir='RESULTS/', mode='eval'):
        """A finite element model for intelligent meta elements.

        Args:
            result_dir:     directory to store results in

        """
        self.result_dir = result_dir
        assert mode in ['train', 'eval']
        self.mode = mode
        self.nodes = Nodes()
        self.elements = Elements(self)
        # Properties
        self.materials = Materials()
        self.properties = Properties()
        # Boundaries
        self.bc = BC(self.nodes)
        # State variables
        self.loads = None
        self.loads_reduce = None
        self.loads_lock = None
        self.loads_external = None
        self.loads_external_reduce = None
        self.loads_external_lock = None
        self.displacements = None
        self.displacements_reduce = None
        self.displacements_lock = None                                          # TODO: isn't this the same as displacements_boundary?
        self.displacements_boundary = None
        self.stiffness = None
        self.stiffness_reduce = None
        self.stiffness_lock = None
        self.stiffness_conter = None
        self.mass = None
        self.mass_reduce = None
        self.mass_lock = None
        self.mass_conter = None
        # Create element class dictionary
        element_classes = [DummyElement]
        self.element_classes = BaseDict(
            zip([c.element_type for c in element_classes], element_classes))
        # Visualization
        self.visualizer = MetaVisualizer(self)
        # Shared neural network model (will be set by elements)
        self.shared_nn_model = None

    def __del__(self):
        if self.shared_nn_model:
            del self.shared_nn_model

    def _all_subclasses(self, cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in self._all_subclasses(c)])

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_elements(self):
        return len(self.elements)

    @property
    def num_dof_per_node(self):
        return self.nodes.num_dof_per_node

    @property
    def node_dofs(self):
        return self.nodes.node_dofs

    @property
    def element_dofs(self):
        return self.elements.element_dofs

    @property
    def num_dofs(self):
        return self.node_dofs.size

    @property
    def dofs(self):
        return self.node_dofs.ravel(order='C')

    @property
    def num_lock_dofs(self):
        return self.bc.num_lock_dofs

    @property
    def num_reduce_dofs(self):
        return self.num_dofs - self.num_lock_dofs

    @property
    def lock_dofs(self):
        return self.bc.lock_dofs

    @property
    def reduce_dofs(self):
        return self.bc.reduce_dofs

    @property
    def zero_state_vector(self):
        return np.zeros([self.num_dofs], dtype=np.float64)

    @property
    def zero_state_matrix(self):
        return np.zeros([self.num_nodes, self.num_dof_per_node],
                        dtype=np.float64)

    @property
    def zero_system_matrix(self):
        return np.zeros([self.num_dofs] * 2, dtype=np.float64)

    @property
    def ie_states(self):
        return [v.fe_state for k, v in self.elements.instances.items()]

    @ie_states.setter
    def ie_states(self, ie_states):
        for l in self.elements.instances.keys():
            self.elements.instances[l].fe_state = ie_states[l - 1]

    def to_state_matrix(self, state_vector_reduce, state_vector_lock):
        state_vector = self.to_state_vector(state_vector_reduce,
                                            state_vector_lock)
        return np.reshape(state_vector, [-1, self.num_dof_per_node], order='C')

    def to_state_vector(self, state_vector_reduce, state_vector_lock):
        reduce_dofs = self.reduce_dofs
        lock_dofs = self.lock_dofs
        state_vector = self.zero_state_vector
        state_vector[reduce_dofs] = state_vector_reduce
        state_vector[lock_dofs] = state_vector_lock
        return state_vector

    def from_state_matrix(self, state_matrix):
        state_vector = np.ravel(state_matrix, order='C')
        return self.from_state_vector(state_vector)

    def from_state_vector(self, state_vector):
        return state_vector[self.reduce_dofs], state_vector[self.lock_dofs]

    def instantiate_element(self, element_def):
        # Extract static element type
        element_type = int(element_def[1])
        if str(element_type).startswith('999'):
            static_element_type = 999
        else:
            static_element_type = element_type
        # Find correct element class
        current_element_class = self.element_classes[static_element_type]
        # Instantiate
        element = current_element_class(self, element_def, mode=self.mode)
        return element

    def generate_beam(self, geometry, grid_shape, element_type, num_elements,
                      material=1, properties=1):
        """Generate a beam model of elements."""
        coords = generate_grid(geometry, grid_shape)
        num_nodes_ie = 1
        for iie in range(num_elements):
            # Element and nodes
            element_node_labels = []
            min_coord = iie * geometry[0]
            max_coord = min_coord + geometry[0]
            for coord in coords + np.asarray([min_coord, 0.0]):
                if (np.any(coord == min_coord) or np.any(coord == max_coord)
                        or np.any(coord == 0.0) or np.any(coord == 1.0)):
                    node_label = self.nodes.add_node(
                        [num_nodes_ie] + coord.tolist())
                    element_node_labels.append(node_label)
                    if node_label == num_nodes_ie:
                        num_nodes_ie += 1
            self.elements.add_element(
                [iie + 1, element_type, material, properties]
                + element_node_labels)

    def _add_nodes_and_element(self, num_nodes_ie, num_elements_ie, coords,
                               element_type, material, properties):
        # Element and nodes
        element_node_labels = []
        min_coord = np.min(coords, axis=0)
        max_coord = np.max(coords, axis=0)
        for coord in coords:
            if (np.any(coord == min_coord)
                    or np.any(coord == max_coord)):
                node_label = self.nodes.add_node(
                    [num_nodes_ie + 1] + coord.tolist())
                element_node_labels.append(node_label)
                if node_label == num_nodes_ie + 1:
                    num_nodes_ie += 1
        self.elements.add_element(
            [num_elements_ie + 1, element_type, material,
             properties] + element_node_labels)
        num_elements_ie += 1
        return num_nodes_ie, num_elements_ie


class BaseVisualizer(object):

    def __init__(self, model):
        # Extract model
        self.model = model
        self.displacements = None
        self.elements = self.model.elements
        self.dirichlet = self.model.bc.dirichlet
        self.neumann = self.model.bc.neumann
        self.locks = self.model.bc.dirichlet.locks
        # plot colors
        self.node_color = 'k'
        self.element_color = 'gray'
        self.element_edgecolor = 'k'
        self.element_alpha = 1.0
        self.dirichlet_color = '0.5'
        self.force_color = 'm'
        self.displacement_color = 'c'
        # plot line widths
        self.element_linewidth = 1.0
        self.bc_linewidth = self.element_linewidth
        # plot sizes
        self.displacement_scale = 1.0
        self.node_size = 3 * self.element_linewidth
        # plot options
        self.node_plotting = True
        self.element_plotting = True
        self.node_label_plotting = False
        self.element_label_plotting = False
        # tolerance
        self.tol = 1e-8

    @property
    def dirichlet_size(self):
        return 0.05 * (np.max(self.model.nodes.node_coords)
                       - np.min(self.model.nodes.node_coords))

    @staticmethod
    def save_figure(path, **kwargs):
        file_type = path.split('.')[-1]
        if file_type in ['tikz', 'tex']:
            BaseVisualizer.export_plot_to_tikz(path, **kwargs)
        else:
            if 'transparent' not in kwargs.keys():
                kwargs['transparent'] = False
            plt.savefig(path, **kwargs)

    @staticmethod
    def export_plot_to_tikz(path, fig=None, in_tikzpicture=True):
        if fig is None:
            fig = plt.gcf()
        height = fig.get_figheight()
        width = fig.get_figwidth()
        tikz_save(path,
                  figurewidth='\\linewidth',
                  figureheight='{:f}\\linewidth'.format(width/height),
                  wrap=in_tikzpicture)

    @staticmethod
    def hide_axes(fig=None):
        if fig == None:
            fig = plt.gcf()
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        for ax in fig.axes:
            ax.axis('off')
        plt.title('')
        fig.set_tight_layout(True)

    def plot_model(self, stress_type=None, color=None, edgecolor=None):
        node_color = edgecolor or self.node_color
        element_color = color or self.element_color
        element_edge_color = edgecolor or self.element_edgecolor
        nodes = self.model.nodes
        if self.displacements is not None:
            displacements = self.displacement_scale * np.asarray(
                self.displacements, dtype=np.float64)
        else:
            displacements = None
        if self.element_plotting:
            tpc = self.draw_elements(
                stress_type=stress_type,  color=element_color,
                edgecolor=element_edge_color)
        else:
            tpc = None
        if self.node_plotting:
            self.draw_nodes(displacements=displacements, color=node_color)
        if self.node_label_plotting:
            self.draw_node_labels(nodes, color=node_color)
        # plt.xlabel('$x$')
        # plt.ylabel('$y$')
        plt.axis('equal')
        return tpc

    # def plot_displacement_field(self, color=None):
    #     nodes = self.model.nodes
    #     nodes_array = nodes.array
    #     if color is None:
    #         color = self.displacement_color
    #     x1 = nodes_array[:, 1]
    #     x2 = nodes_array[:, 2]
    #     u1 = self.displacements[:, 0] * self.disp_scale
    #     u2 = self.displacements[:, 1] * self.disp_scale
    #     self.draw_displacement_field(x1, x2, u1, u2, color=color)
    #     limits = self._calc_plot_limits(nodes.array)
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.axis('equal')
    #     plt.axis(limits)

    def plot_force_field(self, forces=None, color=None, force_scale=1.0):
        nodes = self.model.nodes
        nodes_array = nodes.array
        if forces is None:
            forces = np.zeros_like(nodes.node_dofs, dtype=np.float64)
            neumann = self.neumann
            for node_label, neumann_load in neumann.items():
                forces[node_label - 1, :] = neumann_load[1:]
        if color is None:
            color = self.force_color
        x1 = np.copy(nodes_array[:, 1])
        x2 = np.copy(nodes_array[:, 2])
        if self.displacements is not None:
            displacements = np.asarray(self.displacements, dtype=np.float64)
            x1 += displacements[:, 0] * self.displacement_scale
            x2 += displacements[:, 1] * self.displacement_scale
        f1 = force_scale * forces[:, 0]
        f2 = force_scale * forces[:, 1]
        self.draw_force_field(x1, x2, f1, f2, color=color)
        # limits = self._calc_plot_limits(nodes.array)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.axis('equal')
        # plt.axis(limits)

    @staticmethod
    def _calc_plot_limits(nodes_array):
        max_x = np.max(nodes_array[:, 1])
        min_x = np.min(nodes_array[:, 1])
        plt_width = np.abs(max_x - min_x)
        if plt_width == 0.0:
            plt_width = 1.0
        max_y = np.max(nodes_array[:, 2])
        min_y = np.min(nodes_array[:, 2])
        plt_height = np.abs(max_y - min_y)
        if plt_height == 0.0:
            plt_heigth = 1.0
        # set limits
        x_min = min_x - plt_width * 0.1
        x_max = max_x + plt_width * 0.1
        y_min = min_y - plt_height * 0.1
        y_max = max_y + plt_height * 0.1
        limits = np.asarray([x_min, x_max, y_min, y_max])
        return limits

    def draw_elements(self, stress_type=None, color=None, edgecolor=None):
        raise NotImplementedError

    def draw_nodes(self, displacements=None, color=None):
        if color is None:
            color = self.node_color
        nodes_array = self.model.nodes.array
        x_coord = nodes_array[:, 1]
        y_coord = nodes_array[:, 2]
        if displacements is not None:
            x_coord = x_coord + displacements[:, 0]
            y_coord = y_coord + displacements[:, 1]
        axes = plt.gca()
        max_zorder = max([c.zorder for c in axes.get_children()])
        plt.plot(x_coord, y_coord, ' o', color=color, markersize=self.node_size,
                 zorder=(max_zorder + 1))

    def draw_node_labels(self, nodes, color=None):
        if color is None:
            color = self.node_color
        node_dofs = nodes.node_dofs
        for node_label, node in nodes.items():
            x = node[1]
            y = node[2]
            plt.text(x, y, node_label, color=color)
            if DEBUG:
                xx = x + self.dirichlet_size / 4
                yy = y - self.dirichlet_size / 2
                plt.text(xx, yy,
                         '{:d}\n{:d}'.format(node_dofs[node_label - 1, 0],
                                             node_dofs[node_label - 1, 1]),
                         va='top', ha='left', color='r', size=6)

    def draw_dirichlet_bc(self, dirichlet_color=None, displacement_color=None,
                          hide_field=False, hide_single_dof_constraints=False):
        nodes = self.model.nodes
        locks = self.locks
        if dirichlet_color is None:
            dirichlet_color = self.dirichlet_color
        if displacement_color is None:
            displacement_color = self.displacement_color
        for node_label, lock in locks.items():
            lock_x = int(lock[1])
            lock_y = int(lock[2])
            # x and y locked
            if lock_x == 1 or lock_y == 1:
                if self.displacements is not None:
                    u1 = (self.displacements[node_label - 1, 0]
                          * self.displacement_scale)
                    u2 = (self.displacements[node_label - 1, 1]
                          * self.displacement_scale)
                else:
                    u1 = 0.0
                    u2 = 0.0
                x1 = nodes[node_label][1] + u1
                x2 = nodes[node_label][2] + u2
                if np.abs(u1) < self.tol and np.abs(u2) < self.tol:
                    if self.dirichlet_size:
                        # fixed bearing
                        self._draw_dirichlet_xy_locked(x1, x2,
                                                       color=dirichlet_color)
                elif not (np.abs(u1) < self.tol and np.abs(u2) < self.tol):
                    if self.dirichlet_size:
                        # sliding bearing
                        if (np.abs(u1) < self.tol
                                and not hide_single_dof_constraints):
                            # sliding bearing lock in x
                            self._draw_dirichlet_x_locked(x1, x2,
                                                          color=dirichlet_color)
                        if (np.abs(u2) < self.tol
                                and not hide_single_dof_constraints):
                            # sliding bearing lock in y
                            self._draw_dirichlet_y_locked(x1, x2,
                                                          color=dirichlet_color)
                    # prescribed displacement
                    if not hide_field:
                        self.draw_displacement_field(x1, x2, u1, u2,
                                                     color=displacement_color)

    def _draw_dirichlet_xy_locked(self, x, y, color=None):
        if color is None:
            color = self.dirichlet_color
        size = self.dirichlet_size
        x_coord = np.array([x, x + size * 0.5, x - size * 0.5, x])
        y_coord = np.array([y, y - size * 0.5, y - size * 0.5, y])
        plt.plot(x_coord, y_coord, color=color, linewidth=self.bc_linewidth)

    def _draw_dirichlet_xyr_locked(self, x, y, color=None):
        if color is None:
            color = self.dirichlet_color
        size = self.dirichlet_size
        x_coord = np.array(
            [x + size * 0.5, x + size * 0.5, x - size * 0.5, x - size * 0.5,
             x + size * 0.5])
        y_coord = np.array(
            [y - size * 0.5, y + size * 0.5, y + size * 0.5, y - size * 0.5,
             y - size * 0.5])
        plt.plot(x_coord, y_coord, color=color, linewidth=self.bc_linewidth)

    def _draw_dirichlet_y_locked(self, x, y, color=None):
        if color is None:
            color = self.dirichlet_color
        size = self.dirichlet_size
        x_coord = np.array([x, x + size * 0.5, x - size * 0.5, x])
        y_coord = np.array([y, y - size * 0.5, y - size * 0.5, y])
        plt.plot(x_coord, y_coord, color=color, linewidth=self.bc_linewidth)
        x_line = np.array([x - size * 0.5, x + size * 0.5])
        y_line = np.array([y - size * 0.7, y - size * 0.7])
        plt.plot(x_line, y_line, color=color, linewidth=self.bc_linewidth)

    def _draw_dirichlet_x_locked(self, x, y, color=None):
        if color is None:
            color = self.dirichlet_color
        size = self.dirichlet_size
        x_coord = np.array([x, x - size * 0.5, x - size * 0.5, x])
        y_coord = np.array([y, y + size * 0.5, y - size * 0.5, y])
        plt.plot(x_coord, y_coord, color=color, linewidth=self.bc_linewidth)
        x_line = np.array([x - size * 0.7, x - size * 0.7])
        y_line = np.array([y - size * 0.5, y + size * 0.5])
        plt.plot(x_line, y_line, color=color, linewidth=self.bc_linewidth)

    def draw_displacement_field(self, x, y, u, v, color=None):
        if color is None:
            color = self.displacement_color
        scale = self.displacement_scale
        self.draw_field(x, y, u, v, color=color, pivot='tip', scale=scale)

    def draw_force_field(self, x, y, f1, f2, color=None):
        if color is None:
            color = self.force_color
        coord_size = max(np.abs(np.max(x) - np.min(x)),
                         np.abs(np.max(y) - np.min(y)))
        value_size = max(np.abs(np.max(f1) - np.min(f1)),
                         np.abs(np.max(f2) - np.min(f2)))
        if value_size > 0.0:
            scale = 0.2 * coord_size / value_size
        else:
            scale = 1.0
        self.draw_field(x, y, scale * f1, scale * f2, color=color, pivot='tip',
                        scale=1.0)

    def draw_field(self, x, y, v1, v2, color=None, pivot='tip', scale=1.0):
        max_zorder = max([c.zorder for c in plt.gca().get_children()])
        # Remove zero vectors
        mask = np.logical_or(np.abs(v1) > self.tol, np.abs(v2) > self.tol)
        if mask.any():
            plt.quiver(
                x[mask], y[mask], v1[mask], v2[mask], scale=scale,
                scale_units='xy', pivot=pivot, color=color,
                zorder=(max_zorder + 1))

    def draw_disp_scale(self, x=None, y=None):
        if self.displacement_scale != 1.0:
            if x is None:
                x = 0.5
            if y is None:
                y = 0.95
            plt.text(x, y, 'deformation scale: {:g}'.format(
                self.displacement_scale),
                     transform=plt.gca().transAxes, ha='center', va='top',
                     color='k')


class MetaVisualizer(BaseVisualizer):

    def __init__(self, model):
        super(MetaVisualizer, self).__init__(model)
        self.colormap = 'jet'
        # self.colormap = 'viridis'
        self.climits = None
        self.static_climits = False

    @staticmethod
    def assemble_path(coords):
        coords = np.asarray(coords)
        verts = np.concatenate([coords, coords[0:1, :]], axis=0)
        codes = tuple([Path.MOVETO] + [Path.LINETO] * (verts.shape[0] - 2)
                      + [Path.CLOSEPOLY])
        return Path(verts, codes=codes)

    def draw_elements(self, stress_type=None, displacement_scale=None,
                      color=None, edgecolor=None):
        color = color or self.element_color
        edgecolor = edgecolor or self.element_edgecolor
        displacement_scale = displacement_scale or self.displacement_scale
        elements = self.elements
        if self.climits is None or not self.static_climits:
            # Calculate color limits
            fe_states = self.model.ie_states
            stress_values = [BaseElement.extract_stress_values(
                f['Si'], stress_type=stress_type)
                if (f is not None and f['Si'] is not None)
                else 0.0
                for f in fe_states]
            vmax = max(np.max(stress_values), 0.0)
            vmin = min(np.min(stress_values), 0.0)
            self.climits = [vmin, vmax]
        # Draw elements
        tpc = None
        for element_label, element_def in elements.items():
            if DEBUG:
                print('element_label', element_label)
            # Instantiate element
            element = self.model.elements.instances[element_label]
            tpc = element.draw_element(
                visualizer=self, stress_type=stress_type,
                displacement_scale=displacement_scale, facecolor=color,
                edgecolor=edgecolor, use_global_coords=True)
            element_coords = np.asarray(element.nodes)[:, 1:]
            if self.element_label_plotting:
                xy = np.mean(element_coords, axis=0)
                if (element.fe_state and 'Ui' in element.fe_state.keys()
                        and element.fe_state['Ui'] is not None):
                    uv = np.mean(element.fe_state['Ui'], axis=0)
                    xy += self.displacement_scale * uv
                plt.text(xy[0], xy[1], element_label, ha='center',
                         va='center', color=color)
        if self.climits is not None:
            plt.clim(*self.climits)
        return tpc

    def draw_gradient_patch(self, path, grid, values,
                            edgecolor=None, linewidth=None, alpha=None,
                            colormap=None, shading='gouraud'):
        """Workaround for matplotlib unable to draw gradient filled patches."""
        edgecolor = edgecolor or self.element_edgecolor
        linewidth = linewidth or self.element_linewidth
        alpha = alpha or self.element_alpha
        colormap = colormap or self.colormap
        # Calculate limits
        if not self.climits:
            vmin = min(np.min(values), 0.0)
            vmax = max(np.max(values), 0.0)
        else:
            vmin = self.climits[0]
            vmax = self.climits[1]
        if np.abs(vmax - vmin) < 1e-3:
            vmax = 1.0
            vmin = 0.0
        # Extract current coordinates
        tpc = plt.tripcolor(grid[:, 0], grid[:, 1], values, shading=shading,
                            cmap=colormap, edgecolor='k', alpha=alpha,
                            vmax=vmax, vmin=vmin, linewidth=0.0,
                            antialiased=True, rasterized=True)
        # Create transparent patch for clipping
        axes = plt.gca()
        max_zorder = max([c.zorder for c in axes.get_children()])
        patch = PathPatch(path, facecolor='None', edgecolor=edgecolor,
                          linewidth=linewidth, zorder=(max_zorder + 3),
                          rasterized=True)
        axes.add_patch(patch)
        tpc.set_clip_path(patch)
        return tpc

    def draw_uniform_patch(self, path, facecolor=None, edgecolor=None,
                           linewidth=None, alpha=None):
        facecolor = facecolor or self.element_color
        edgecolor = edgecolor or self.element_edgecolor
        linewidth = linewidth or self.element_linewidth
        alpha = alpha or self.element_alpha
        # Draw patches (both are necessary!!!)
        axes = plt.gca()
        max_zorder = max([c.zorder for c in axes.get_children()])
        fill_patch = PathPatch(path, facecolor=facecolor, alpha=alpha,
                               edgecolor='None', linewidth=linewidth,
                               zorder=(max_zorder + 1))
        contour_patch = PathPatch(path, facecolor='None', alpha=1.0,
                                  edgecolor=edgecolor, linewidth=linewidth,
                                  zorder=(max_zorder + 2))
        axes.add_patch(fill_patch)
        axes.add_patch(contour_patch)

    def draw_internal_grid(self, vertices, connectivity, edgecolor=None,
                           linewidth=None):
        edgecolor = edgecolor or self.element_edgecolor
        linewidth = linewidth or self.element_linewidth
        # Loop over all elements and create patches
        for con in connectivity:
            path_coords = vertices[con, :]
            path = grids.assemble_path(path_coords)
            axes = plt.gca()
            max_zorder = max([c.zorder for c in axes.get_children()])
            patch = PathPatch(path, facecolor='None',
                              edgecolor=edgecolor,
                              linewidth=linewidth / 2.0,
                              zorder=(max_zorder + 3))
            axes.add_patch(patch)


class DataReader(object):

    def __init__(self, data_shape, data_format, batch_size,
                 input_features=None, output_features=None, src_type='file',
                 sample_dtype=tf.float64, cast_dtype=tf.float64,
                 slice_tensors=False, file_type='tfr', tfr_features=None,
                 csv_skip=0, csv_comment='#', num_epochs=None, shuffle=True,
                 prefetch=True, buffer_size=10000, chunk_size=None,
                 iter_probability=0.0, seed=None):
        """A class that loads and batches data into tensorflow tensors.

        DataReader uses the new tensorflow dataset API to load files into
        tf.Dataset objects. The class currently supports reading of csv and
        tfrecord files. When processing tfrecords and no tfr_features is present
        it defaults to the cids default tfrecord format (see below).

        Args:
            data_shape:     Shape of the input data
            data_format:    Data format used ('NF', NSF', or 'NCHW')
                                 N:             batch axis,
                                 S:             sequence axis
                                 H/W/D/X/Y/Z:   spatial axes
                                 F/C:           feature or channel axis
            batch_size:     The desired batch size (accepts dicts)
            src_type:       The source of data ('file' or 'placeholder')
            sample_dtype:   Data type in the data file
            cast_dtype:     Data type to cast to
            slice_tensors:  Boolean: Slice dataset into multiple samples?
            tfr_features:   Dictionary of protobuf keys and tf.Features()
                            (Defaults to cids default format) for tfrecords
            csv_skip:       Number of lines to skip for csv
            csv_comment:    Lines to skip starting with this string for csv
            num_epochs:     Number of epochs to repeat data (None for infinite)
            shuffle:        Boolean: Shuffle dataset using buffer_size?
            prefetch:       Boolean: Preload dataset of buffer_size?
            buffer_size:    Number of samples to collect for shuffling
            chunk_size:     Size of chunks along sequence axis (None: disabled)
            seed:           Set a seed for shuffling. (Default None: disabled)
        """
        if tfr_features is None:
            # Use default cids tfrecord format
            self.tfr_features = {'data':
                                     tf.io.FixedLenFeature(
                                         (), tf.string, default_value=''),
                                 'data_shape':
                                     tf.io.FixedLenFeature(
                                         (len(data_shape),), tf.int64)}
        else:
            self.tfr_features = tfr_features
        self.input_features = input_features
        self.output_features = output_features
        self.data_shape = data_shape
        if isinstance(batch_size, int):
            self.batch_size = batch_size
            self.valid_batch_size = batch_size
            self.test_batch_size = batch_size
        elif hasattr(batch_size, 'keys'):
            self.batch_size = batch_size['train']
            self.valid_batch_size = batch_size['valid']
            try:
                self.test_batch_size = batch_size['test']
            except KeyError:
                self.test_batch_size = batch_size['valid']
        else:
            raise ValueError('Variable batch_size must be type int or dict.')
        self.src_type = src_type
        self.data_format = data_format
        axes = read_axes(data_format)
        self.batch_axis = axes[0]
        self.feature_axis = axes[1]
        self.sequence_axis = axes[2]
        self.spatial_axes = axes[3]
        self.iter_axis = axes[4]
        self.sample_dtype = sample_dtype
        self.cast_dtype = cast_dtype
        self.slice_tensors = slice_tensors
        self.skip = csv_skip
        self.comment = csv_comment
        self.file_type = file_type
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch = prefetch
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.iter_probability = iter_probability
        self.VERBOSITY = 2

    def _parse_tfrecord(self, tfr):
        parsed = tf.io.parse_single_example(tfr, self.tfr_features)
        data = parsed['data']
        dynamic_data_shape = parsed['data_shape']
        # Decode and cast
        data = tf.io.decode_raw(data, self.sample_dtype)
        data = tf.cast(data, self.cast_dtype)
        dynamic_data_shape = tf.cast(dynamic_data_shape, tf.int32)
        # Restore data
        data = tf.reshape(data, dynamic_data_shape)
        data = tf.cond(
            tf.equal(dynamic_data_shape[self.batch_axis], 1),
            lambda: tf.squeeze(data, axis=0), lambda: data)
        # Tile parameter data
        param_keys = sorted([k for k in self.tfr_features.keys()
                             if 'data' not in k])
        if self.VERBOSITY > 2:
            if param_keys:
                print('  Expanding data with parameter features ',
                      '(sorted: alphanumerically):')
                print('    ', param_keys)
        param_data = [parsed[k] for k in param_keys]
        # Cast
        param_data = [tf.cast(d, self.cast_dtype) for d in param_data]
        # Expand and tile
        param_data = [tf.squeeze(d) for d in param_data]
        for p in param_data:
            assert len(p.shape) == 0,\
                'All parameters without "data" in their key must be scalars.'
        active_feature_axis = self.feature_axis
        active_dims = list(range(len(self.data_shape)))
        if not self.slice_tensors:
            active_dims.remove(self.batch_axis)
        for _ in active_dims:
            param_data = [tf.expand_dims(d, 0) for d in param_data]
        # Tile (replace feature size with 1)
        tile_shape = tf.gather(dynamic_data_shape, active_dims)
        tile_shape = tf.unstack(tile_shape)
        tile_shape[active_feature_axis] = tf.ones([], dtype=tf.int32)
        tile_shape = tf.stack(tile_shape)
        param_data = [tf.tile(d, tile_shape) for d in param_data]
        # Concat
        data = tf.concat([data] + param_data, axis=active_feature_axis)
        return data

    def _parse_csv_line(self, line):
        if self.cast_dtype == tf.float32 or self.cast_dtype == tf.float64:
            defaults = [[0.0]] * self.data_shape[-1]
        elif self.cast_dtype == tf.int64 or self.cast_dtype == tf.int32:
            defaults = [[0]] * self.data_shape[-1]
        else:
            raise ValueError('Unknown self.cast_dtype = ', self.cast_dtype)
        parsed = tf.decode_csv(line, record_defaults=defaults)
        parsed = tf.stack(parsed, axis=0)
        return parsed

    def _parse_image_from_tfr(self, example):
        # TODO: features in class
        features = tf.parse_single_example(
            example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        label = tf.cast(features['label'], tf.int32)
        image = tf.image.decode_image(features['image_raw'])
        return image, label

    def _parse_image_and_target_from_tfr(self, example):
        # TODO: melt with other functions?!
        features = tf.parse_single_example(example,
                                           features={
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'target_raw': tf.FixedLenFeature([], tf.string),
                                               'n_target': tf.FixedLenFeature([], tf.int64)
                                           })
        image = tf.image.decode_image(features['image_raw'])
        target = tf.decode_raw(features['target_raw'], self.sample_dtype)
        target = tf.cast(target, self.cast_dtype)
        n_target = tf.cast(features['n_target'], tf.int32)
        # TODO: reshape!!!
        # target = tf.reshape(target, [n_target, -1])
        return image, target

    def _parse_h5_from_tfr(self, example):
        # TODO: melt with other functions?!
        features = tf.parse_single_example(example, features=self.tfr_features)
        # FIXME: Make flexible
        num_features = tf.cast(features['num_features'], tf.int32)
        velocity = features['velocity']
        weight = features['weight']
        height = features['height']
        length_thigh = features['length_thigh']
        length_shank = features['length_shank']
        length_foot = features['length_foot']
        data = tf.decode_raw(features['data'], self.sample_dtype)
        data = tf.cast(data, self.cast_dtype)
        data = tf.reshape(data, [-1, num_features])
        # Concat with context
        context = tf.stack(
            [velocity, weight, height, length_thigh, length_shank, length_foot])
        context_data = tf.expand_dims(context, axis=0)
        context_tiled = tf.tile(context_data, [tf.shape(data)[0], 1])
        data = tf.concat([data, context_tiled], axis=-1)
        return data

    def generate_batch_dataset(self, data, mode='train'):
        """Read data as specified by src_type during object initialisation.

        Args:
            data:       Data source tensor
            mode:       Which mode to use (defines batch sized used)

        Returns:
            dataset:    A tensorflow Dataset (for iterator)

        """
        if self.src_type[:5].lower() == 'file':
            dataset = self.generate_dataset_from_files(data)
        elif self.src_type[:5].lower() == 'place':
            dataset = self.generate_dataset_from_placeholders(data)
        else:
            raise ValueError(
                'Unknown src_type: must be either "file" or "placeholder".')
        # Define final shape of sample
        sample_shape = list(deepcopy(self.data_shape))
        del sample_shape[self.batch_axis]
        # Chunk sequence dataset
        if self.sequence_axis is not None and self.chunk_size:
            assert self.data_shape[self.sequence_axis] is not None, \
                'Data shape along sequence axis must be defined when ' \
                'chunk_size is set.'
            sample_sequence_axis = self.sequence_axis - 1
            sequence_length = sample_shape[sample_sequence_axis]
            # Shuffle dataset before synthesizing more samples
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=self.buffer_size,
                                          seed=self.seed)
            # Insure that all sequences are padded with zeros to max length
            dataset = dataset.padded_batch(batch_size=1,
                                           padded_shapes=sample_shape)
            dataset = dataset.map(tf.squeeze)  # remove added axis
            # Create multiple sequence chunks from chunkable sequences
            if sequence_length > self.chunk_size:
                sample_shape[sample_sequence_axis] = self.chunk_size
                num_chunks = sequence_length - self.chunk_size
                chunk_idx = [list(range(i, i + self.chunk_size))
                             for i in range(num_chunks)]
                dataset = dataset.flat_map(
                    lambda sample: tf.data.Dataset.from_tensor_slices(
                        [tf.gather(sample, c, axis=sample_sequence_axis)
                         for c in chunk_idx]))
                # remove blank samples
                dataset = dataset.filter(
                    lambda sample: tf.reduce_any(tf.cast(
                        tf.sign(tf.abs(sample)), dtype=tf.bool)))
        # Select random iteration from iteration axis
        if self.iter_axis is not None:
            sample_iter_axis = self.iter_axis - 1
            if self.iter_probability > 0.0:
                # Insure that all sequences are padded with zeros to max length
                dataset = dataset.padded_batch(batch_size=1,
                                               padded_shapes=sample_shape)
                dataset = dataset.map(tf.squeeze)  # remove added axis
                # Take random iteration sample (index > 0)
                dataset = dataset.map(self._extract_random_iterations)
                del sample_shape[sample_iter_axis]
            else:
                # Take only converged sample (index 0)
                dataset = dataset.map(
                    lambda sample: tf.squeeze(
                        tf.gather(sample, [0], axis=sample_iter_axis),
                        axis=sample_iter_axis))
                del sample_shape[sample_iter_axis]
        # # Cache
        # if self.prefetch:
        #     dataset = dataset.prefetch(buffer_size=self.buffer_size)
        # Shuffle dataset after synthesizing samples
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size,
                                      seed=self.seed)
        # Repeat dataset
        if self.num_epochs is not None:
            dataset = dataset.repeat(self.num_epochs)
        elif self.num_epochs != 0:
            dataset = dataset.repeat()
        # Create batches
        if mode == 'valid':
            batch_size = self.valid_batch_size
        else:
            batch_size = self.batch_size
        dataset = dataset.padded_batch(
            batch_size=batch_size, padded_shapes=sample_shape)
        if self.input_features is not None:
            dataset = dataset.map(
                lambda batch: (tf.gather(batch, self.input_features,
                                         axis=self.feature_axis),
                               tf.gather(batch, self.output_features,
                                         axis=self.feature_axis)))
        return dataset

    def _get_random_iter(self, frame):
        # TODO: hard coded iter axis
        # Get all axes but sequence and iteration
        axes = tuple([i for i, d in enumerate(self.data_format[2:])
                      if d not in 'I'])
        # Find mask of iterations with non-zero features for all frames
        iter_mask = tf.sign(tf.reduce_max(tf.abs(frame), axis=axes))
        # Count numbers of non-zero iterations for each frame
        iter_length = tf.reduce_sum(iter_mask)
        iter_length = tf.cast(iter_length, tf.int32)
        iter_length = tf.maximum(iter_length, 1)
        # Which iteration to choose
        iter_index = tf.random.uniform([], 0, iter_length, dtype=tf.int32)
        iter_index *= tf.cast(
            tf.keras.backend.random_binomial([], self.iter_probability),
            dtype=tf.int32)
        return frame[..., 0, :] # FIXME

    def _extract_random_iterations(self, sample):
        # Get selected iteration for each frame in sequence
        sample_sequence_axis = self.sequence_axis - 1
        frames = tf.unstack(sample, num=self.data_shape[self.sequence_axis],
                            axis=sample_sequence_axis)
        frames = [self._get_random_iter(frame) for frame in frames]
        sample = tf.stack(frames, axis=sample_sequence_axis)
        return sample

    def generate_test_dataset(self, data, batch_size=None):
        """Create a dataset of tensors from various data sources.

        Args:
            data:          List of data
        Returns:
            dataset:        A dataset of tensors
        """
        if batch_size is None:
            batch_size = self.test_batch_size
        if self.src_type[:5].lower() == 'file':
            dataset = self.generate_dataset_from_files(data)
        elif self.src_type[:5].lower() == 'place':
            dataset = self.generate_dataset_from_placeholders(data)
        else:
            raise ValueError(
                'Unknown src_type: must be either "file" or "placeholder".')
        # Define final shape of sample
        sample_shape = list(deepcopy(self.data_shape))
        del sample_shape[self.batch_axis]
        # Select converged sample if iteration axis is present
        if self.iter_axis is not None:
            sample_iter_axis = self.iter_axis - 1
            # Take only converged sample (index 0)
            dataset = dataset.map(
                lambda sample: tf.squeeze(
                    tf.gather(sample, [0], axis=sample_iter_axis),
                    axis=sample_iter_axis))
            del sample_shape[sample_iter_axis]
        # Create batch of the entire dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size, padded_shapes=sample_shape)
        if self.input_features is not None:
            dataset = dataset.map(
                lambda batch: (tf.gather(batch, self.input_features,
                                         axis=self.feature_axis),
                               tf.gather(batch, self.output_features,
                                         axis=self.feature_axis)))
        return dataset

    def generate_dataset_from_files(self, files):
        """Create a dataset of tensors from files.

        Args:
            files:          List of filenames
        Returns:
            dataset:    A dataset of tensors
        """
        # Parse file
        if self.file_type[:3] == 'tfr':
            dataset = tf.data.TFRecordDataset(files)
            dataset = dataset.map(self._parse_tfrecord)
            if self.slice_tensors:
                dataset = dataset.flat_map(
                    lambda d: tf.data.Dataset.from_tensor_slices(d))
        elif self.file_type[:3] == 'csv':
            dataset = tf.data.Dataset.from_tensor_slices(files)
            # Skip line and filter out comments
            dataset = dataset.flat_map(
                lambda filename: (
                    tf.data.TextLineDataset(filename)
                        .skip(self.skip)
                        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1),
                                                          self.comment))))
            dataset = dataset.map(self._parse_csv_line)
            if not self.slice_tensors:
                # Stack sequence
                if self.sequence_axis:
                    dataset = dataset.batch(
                        batch_size=self.data_shape[self.sequence_axis])
        else:
            raise ValueError('file_type "{:s}" unknown.'.format(self.file_type))
        return dataset

    def generate_dataset_from_placeholders(self, placeholders):
        """Create a dataset of tensors from placeholders.

        Args:
            placeholders:   A tensorflow placeholder for feeding
        Returns:
            dataset:        A dataset of tensors
        """
        dataset = tf.data.Dataset.from_tensors(placeholders)
        if self.slice_tensors:
            dataset = dataset.flat_map(
                lambda d: tf.data.Dataset.from_tensor_slices(d))
        dataset = dataset.map(
            lambda sample: tf.cast(sample, dtype=self.cast_dtype))
        return dataset

    def read_tfrecords(self, tfr_files):
        """Reads a single tfrecord file to check its content."""
        dataset = self.generate_test_dataset(tfr_files)
        samples = [s[0].numpy() for s in dataset]
        if len(tfr_files) == 1:
            samples = np.squeeze(samples, axis=self.batch_axis)
        else:
            samples = np.concatenate(samples, axis=0)
        return samples
