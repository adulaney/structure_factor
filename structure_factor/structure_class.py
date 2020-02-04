"""
This is a package of analysis functions that calculate the intermediate scattering
function (ISF) from a GSD simulation trajectory file.
"""
import pandas as pd
import numpy as np
import cmath
import gsd.hoomd
import itertools
import sys
import structure_factor
from structure_factor import _structure_factor


class _struc_factor:
    r""" Class for computing static structure factor (SSF).

    Args: 
        file (str): Name of GSD trajectory file to compute SSF.

    :py:class:`ssf` computes the static structure factor for a given GSD snapshot.

    """

    # def __init__(self):
    #     self.
    #     return

    def save_file(self, save_path: str):
        save_location = self.type + save_path + ".npy"

        if hasattr(self, "SF"):
            self.SF.to_pickle(save_location)
        else:
            raise RuntimeError("No compute has been performed yet.")


class ssf(_struc_factor):
    r""" Static Structure Factor calculation.

    Args: 
        traj (gsd.hoomd.HOOMDTrajectory): GSD Trajectory

    :py:class:`ssf` computes the static structure factor for a given GSD snapshot.

    """

    def __init__(self, traj, compute_mode="gpu"):
        # Initialize base class
        _struc_factor.__init__(self)

        # Initialize trajectory
        self.traj = traj

        # initialize the reflected c++ class
        self.compute_mode = compute_mode

        if len(self.traj) > 1:
            self.traj_type = "traj"
        else:
            self.traj_type = "frame"

    def compute(self, frame=0, max_kint=15, single_vec=True, gpu_id=0):

        if self.traj_type == "traj":
            self.frame = self.traj[frame]
        else:
            self.frame = self.traj[0]

        # Define box parameters
        Lx = self.frame.configuration.box[0]
        Ly = self.frame.configuration.box[1]

        # Define k_x values
        kxs = 2 * np.pi / Lx * np.arange(0, max_kint + 1).astype(int)

        # Create both integer vectors
        if single_vec is True:
            kys = np.array([0.0])
        else:
            kys = 2 * np.pi / Ly * np.arange(0, max_kint + 1).astype(int)

        # Define value pairs
        k_vecs = list(itertools.product(kxs, kys))

        # Assign magnitude of wavevectors
        if hasattr(self, "k_mag") is False:
            self.k_mag = np.linalg.norm(np.array(k_vecs), axis=1)[1:]

        # Create list of k_vec tuples
        k_vecs = [tup + (0.0,) for tup in k_vecs][1:]

        # Instantiate cpp class
        if self.compute_mode == "gpu":
            self.cpp_method = _structure_factor.StaticStructureFactorGPU(
                self.frame.particles.position.tolist(), k_vecs, gpu_id
            )
        else:
            self.cpp_method = _structure_factor.StaticStructureFactor(
                self.frame.particles.position.tolist(), k_vecs
            )
        struc_val = self.cpp_method.compute()
        self.ssf = np.array(struc_val)

    def bin_avg(self, bin_size: float = 0.1):
        self.bin_size = bin_size
        max_kval = self.k_mag[-1]
        n_bin = np.round(max_kval / self.bin_size)
        k_bins = np.linspace(0, self.bin_size * n_bin, int(n_bin) + 1)

        self.k_bins = k_bins[1:]
        self.ssf_bin = np.zeros(len(self.k_bins))

        for i, k_val in enumerate(k_bins[1:]):
            inds = np.argwhere((self.k_mag <= k_val) & (self.k_mag > k_bins[i]))[:, 0]
            self.ssf_bin[i] = np.mean(self.ssf[inds])


class dsf(_struc_factor):
    r""" Dynamic Structure Factor calculation.

    Args: 
        traj (gsd.hoomd.HOOMDTrajectory): GSD Trajectory

    :py:class:`isf` computes the incoherent dynamic structure factor for a given GSD snapshot.

    """

    def __init__(self, traj, compute_mode="gpu"):
        # Initialize base class
        _struc_factor.__init__(self)

        # Initialize trajectory
        self.traj = traj

        self.compute_mode = compute_mode

        # initialize the reflected c++ class

        if len(self.traj) > 1:
            self.traj_type = "traj"
        else:
            self.traj_type = "frame"

    def compute(self, frame=0, init_frame=0, max_kint=15, single_vec=True, gpu_id=0):

        if self.traj_type == "traj":
            self.frame = self.traj[frame]
            self.init_frame = self.traj[init_frame]
        else:
            self.frame = self.traj[0]
            self.init_frame = self.traj[0]

        # Define box parameters
        Lx = self.frame.configuration.box[0]
        Ly = self.frame.configuration.box[1]

        # Define k_x values
        kxs = 2 * np.pi / Lx * np.arange(0, max_kint + 1).astype(int)

        # Create both integer vectors
        if single_vec is True:
            kys = np.array([0.0])
        else:
            kys = 2 * np.pi / Ly * np.arange(0, max_kint + 1).astype(int)

        # Define value pairs
        k_vecs = list(itertools.product(kxs, kys))

        # Assign magnitude of wavevectors
        if hasattr(self, "k_mag") is False:
            self.k_mag = np.linalg.norm(np.array(k_vecs), axis=1)[1:]

        # Create list of k_vec tuples
        k_vecs = [tup + (0.0,) for tup in k_vecs][1:]

        # Instantiate cpp class
        if self.compute_mode == "gpu":
            self.cpp_method = _structure_factor.DynamicStructureFactorGPU(
                self.init_frame.particles.position.tolist(),
                self.frame.particles.position.tolist(),
                k_vecs,
                gpu_id,
            )
        else:
            self.cpp_method = _structure_factor.DynamicStructureFactor(
                self.init_frame.particles.position.tolist(),
                self.frame.particles.position.tolist(),
                k_vecs,
            )

        dsf = self.cpp_method.compute()
        self.re_dsf = np.array(dsf)[:, 0]
        self.im_dsf = np.array(dsf)[:, 1]

    def bin_avg(self, bin_size: float = 0.1):
        self.bin_size = bin_size
        max_kval = self.k_mag[-1]
        n_bin = np.round(max_kval / self.bin_size)
        k_bins = np.linspace(0, self.bin_size * n_bin, int(n_bin) + 1)

        self.k_bins = k_bins[1:]
        self.re_dsf_bin = np.zeros(len(self.k_bins))
        self.im_dsf_bin = np.zeros(len(self.k_bins))

        for i, k_val in enumerate(k_bins[1:]):
            inds = np.argwhere((self.k_mag <= k_val) & (self.k_mag > k_bins[i]))[:, 0]
            self.re_dsf_bin[i] = np.mean(self.re_dsf[inds])
            self.im_dsf_bin[i] = np.mean(self.im_dsf[inds])
