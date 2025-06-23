"""Routines for augmenting IMU trajectories by scaling and shifting."""

import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.transform import Rotation as R
from .preprocessing import remove_jumps_from_euler_angle
from .constants import INTRINSIC_EULER_ORDER, QUATERNION_AXES



def add_jumps_to_euler(angle_series: np.ndarray) -> np.ndarray:
    """Wrap Euler angles back into the range ``[-180°, 180°]``.

    Parameters
    ----------
    angle_series : np.ndarray
        Euler angle sequence to adjust.

    Returns
    -------
    np.ndarray
        Euler angles wrapped into ``[-180°, 180°]``.
    """
    angle_series = np.where(angle_series < -180, angle_series + 360, angle_series)
    return np.where(angle_series > 180, angle_series - 360, angle_series)


def draw_deltas_from_distribution(
    distribution_parameters: Dict,
    imu: str,
    label: str,
) -> np.ndarray:
    """Sample delta angles for one IMU from a multivariate normal distribution.

    Parameters
    ----------
    distribution_parameters : dict
        Nested dictionary of ``mean_deltas`` and ``cov_deltas`` values.
    imu : str
        Name of the IMU.
    label : str
        Target class label.

    Returns
    -------
    np.ndarray
        Vector of shape ``(3,)`` with sampled deltas for X, Y and Z axes.
    """
    means = distribution_parameters[label][imu]['mean_deltas']
    cov = distribution_parameters[label][imu]['cov_deltas']
    return np.random.multivariate_normal(mean=means, cov=cov)


def draw_offsets_from_distribution(
    distribution_parameters: Dict,
    imu: str,
    label: str,
) -> np.ndarray:
    """Sample offset angles for one IMU from a multivariate normal distribution.

    Parameters
    ----------
    distribution_parameters : dict
        Nested dictionary of ``mean_offsets`` and ``cov_offsets`` values.
    imu : str
        Name of the IMU.
    label : str
        Target class label.

    Returns
    -------
    np.ndarray
        Vector of shape ``(3,)`` with sampled offsets for X, Y and Z axes.
    """
    means = distribution_parameters[label][imu]['mean_offsets']
    cov = distribution_parameters[label][imu]['cov_offsets']
    return np.random.multivariate_normal(mean=means, cov=cov)


def scale_and_shift_euler_angles(
    angle_series: np.ndarray,
    target_delta: float,
    target_offset: float,
) -> np.ndarray:
    """Scale and shift a single Euler angle series to a new range and offset.

    Parameters
    ----------
    angle_series : np.ndarray
        Original Euler angles (1D array of shape ``(T,)``).
    target_delta : float or None
        Desired range of motion. If ``None`` no scaling is applied.
    target_offset : float or None
        Desired starting angle. If ``None`` the original offset is preserved.

    Returns
    -------
    np.ndarray
        Transformed Euler angles.
    """
    angle_series = np.asarray(angle_series)
    adjusted_angles = angle_series.copy()
    original_delta = angle_series.max() - angle_series.min()
    original_offset = angle_series[0]

    # Scaling
    if target_delta is not None:
        scale = target_delta / original_delta if original_delta != 0 else 1.0
        adjusted_angles = (adjusted_angles - original_offset) * scale
    else:
        adjusted_angles = adjusted_angles - original_offset

    # Shifting
    if target_offset is not None:
        adjusted_angles += target_offset
    else:
        adjusted_angles += original_offset

    return adjusted_angles



def augment_sample(
    input_sample,
    imu_names: List[str],
    distribution_parameters: Dict,
    target_label: str,
    no_scaling: bool = False,
    no_shifting: bool = False
):
    """Augment a single IMU sample by random scaling and shifting.

    Parameters
    ----------
    input_sample : pandas.DataFrame
        DataFrame with IMU quaternions for one sample.
    imu_names : list[str]
        List of IMU identifiers.
    distribution_parameters : dict
        Nested dictionary with mean and covariance values for each IMU and
        label.
    target_label : str
        The target label for augmentation.
    no_scaling : bool, optional
        If ``True``, skip scaling. Default is ``False``.
    no_shifting : bool, optional
        If ``True``, skip shifting. Default is ``False``.

    Returns
    -------
    pandas.DataFrame
        The augmented dataframe with updated quaternion columns.
    """

    augmented_sample = input_sample.copy()

    for imu in imu_names:
        # Step 1: Extract intrinsic Euler angles
        quat_cols = [f"{axis}_{imu}" for axis in QUATERNION_AXES]
        euler_angles = R.from_quat(augmented_sample[quat_cols]).as_euler(INTRINSIC_EULER_ORDER, degrees=True)

        # Step 2: Sample deltas and offsets
        if not no_scaling:
            target_deltas = draw_deltas_from_distribution(distribution_parameters, imu, target_label)
        else:
            target_deltas = np.array([None]*3, dtype=object)

        if not no_shifting:
            target_offsets = draw_offsets_from_distribution(distribution_parameters, imu, target_label)
        else:
            target_offsets = np.array([None]*3, dtype=object)

        # Step 3: Apply per-axis transformation
        transformed_euler = np.zeros_like(euler_angles)
        for axis_idx in range(3):
            axis_series = remove_jumps_from_euler_angle(euler_angles[:,axis_idx])
            adjusted = scale_and_shift_euler_angles(axis_series, target_deltas[axis_idx], target_offsets[axis_idx])
            transformed_euler[:, axis_idx] = add_jumps_to_euler(adjusted)

        # Step 4: Convert back to quaternion
        transformed_quat = R.from_euler(INTRINSIC_EULER_ORDER, transformed_euler, degrees=True).as_quat()
        augmented_sample[quat_cols] = transformed_quat

    return augmented_sample










