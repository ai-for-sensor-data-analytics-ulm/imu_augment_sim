"""Pre-processing utilities for IMU quaternion streams."""

import numpy as np
import copy
import pandas as pd
from typing import List, Tuple, Dict
from scipy.spatial.transform import Rotation as R
from .constants import INTRINSIC_EULER_ORDER, QUATERNION_AXES

__all__ = [
    "calibrate_segment",
    "calibrate_segments",
    "transform_quat_to_intrinsic_euler",
    "remove_jumps_from_euler_angle",
    "calculate_distribution_parameters_from_data",
]


def calibrate_segment(
    quaternions: np.ndarray,
    correction_rotation: np.ndarray,
) -> np.ndarray:
    """
    Apply heading correction to a sequence of quaternions using a correction quaternion.

    Parameters
    ----------
    quaternions : np.ndarray, shape (T, 4)
        Quaternion sequence to be corrected.
    correction_rotation : np.ndarray, shape (4,)
        Quaternion representing the rotation to apply.

    Returns
    -------
    np.ndarray
        Calibrated quaternion sequence of shape (T, 4).
    """
    q_orig = R.from_quat(quaternions)
    q_corr = R.from_quat(correction_rotation)
    q_calibrated = q_orig * q_corr
    return q_calibrated.as_quat()


def calibrate_segments(
    data: List[pd.DataFrame],
    imu_names: List[str],
    correction_rotations: Dict[str, np.ndarray]
) -> List[pd.DataFrame]:
    """
    Calibrate quaternion angle_series for multiple IMUs across multiple samples.

    Parameters
    ----------
    data : list of pd.DataFrame
        Each DataFrame contains quaternion columns for a sample.
    imu_names : list of str
        Names of IMUs to calibrate.
    correction_rotations : dict
        Dictionary mapping each IMU to its correction quaternion (shape: (4,)).

    Returns
    -------
    list of pd.DataFrame
        Updated list with calibrated quaternion columns.
    """
    for sample in data:
        for imu in imu_names:
            quat_cols = [f'{axis}_{imu}' for axis in QUATERNION_AXES]
            quat = sample[quat_cols].to_numpy()
            calibrated = calibrate_segment(quat, correction_rotations[imu])
            sample[quat_cols] = calibrated
    return data


def transform_quat_to_intrinsic_euler(
    quats: pd.DataFrame,
    imu_pos_name: Dict[str, str]
) -> pd.DataFrame:
    """
    Transform quaternion columns into intrinsic Euler angles for multiple IMUs.

    Parameters
    ----------
    quats : pd.DataFrame
        DataFrame containing quaternion columns.
    imu_pos_name : dict
        Mapping from position name to IMU name or list of possible names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Euler angles as additional columns.
    """
    intrinsic_euler_angles = pd.DataFrame()
    for pos, imu_name in imu_pos_name.items():
        if isinstance(imu_name, list):
            for i in imu_name:
                if 'w_' + i in quats.columns:
                    imu_name = i
                    break
        col_names = [f"{axis}_{imu_name}" for axis in QUATERNION_AXES]
        rot = R.from_quat(quats[col_names])
        euler_cols = [f"{imu_name}_{axis}" for axis in ['Ox', 'Oy', 'Oz']]
        intrinsic_euler_angles[euler_cols] = rot.as_euler(INTRINSIC_EULER_ORDER, degrees=True)
    return intrinsic_euler_angles


def remove_jumps_from_euler_angle(
    angle_series: np.ndarray,
) -> np.ndarray:
    """
    Remove discontinuities (jumps > 330°) from Euler angle sequences.

    Parameters
    ----------
    angle_series : np.ndarray
        Array of shape (T,) with Euler angle values.

    Returns
    -------
    np.ndarray
        Smoothed angle array with jumps corrected.
    """
    t_diff = np.diff(angle_series, n=1)
    jump_indices = np.argwhere(np.abs(t_diff) > 330).flatten()
    corrected = np.copy(angle_series)

    for i in range(0, len(jump_indices), 2):
        delta = t_diff[jump_indices[i]]
        start = jump_indices[i] + 1
        if (i + 1) > (len(jump_indices) - 1):
            correction = -360 if delta > 0 else 360
            corrected[start:] += correction
        else:
            end = jump_indices[i + 1] + 1
            correction = -360 if delta > 0 else 360
            corrected[start:end] += correction

    return corrected


def calculate_distribution_parameters_from_data(
    data: List[pd.DataFrame],
    imu_names: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Calculate mean and covariance of Euler angle offsets and deltas per IMU and class label.

    Parameters
    ----------
    data : list of pd.DataFrame
        IMU samples, each as a DataFrame with quaternion columns.
    imu_names : list of str
        IMU identifiers (used to extract quaternion columns).
    labels : list of str
        Class labels corresponding to each sample.

    Returns
    -------
    dict
        Nested dict[label][imu] with:
            - 'mean_deltas': np.ndarray, shape (3,)
            - 'cov_deltas' : np.ndarray, shape (3, 3)
            - 'mean_offsets': np.ndarray, shape (3,)
            - 'cov_offsets' : np.ndarray, shape (3, 3)
    """
    # Step 1: Prepare
    axis_labels = ['X', 'Y', 'Z']
    imu_axes = [f'{imu}_{axis}' for imu in imu_names for axis in axis_labels]
    stats = {k: [] for k in imu_axes}

    # Step 2: Collect individual stats per recording
    for sample, label in zip(data, labels):
        for imu in imu_names:
            # Convert quaternion to Euler angles
            quat_cols = [f"{axis}_{imu}" for axis in QUATERNION_AXES]
            euler = R.from_quat(sample[quat_cols]).as_euler(INTRINSIC_EULER_ORDER, degrees=True)

            for idx, axis in enumerate(axis_labels):
                angle_series = remove_jumps_from_euler_angle(euler[:, idx])
                stats[f"{imu}_{axis}"].append({
                    'delta': angle_series.max() - angle_series.min(),
                    'offset': angle_series[0],
                    'label': label,
                })

    # Step 3: Aggregate per label
    distribution_parameters = {}
    for label in set(labels):
        distribution_parameters[label] = {}
        for imu in imu_names:
            deltas = np.array([
                [entry['delta'] for entry in stats[f"{imu}_{axis}"] if entry['label'] == label]
                for axis in axis_labels
            ])

            offsets = np.array([
                [entry['offset'] for entry in stats[f"{imu}_{axis}"] if entry['label'] == label]
                for axis in axis_labels
            ])

            distribution_parameters[label][imu] = {
                'mean_deltas': np.mean(deltas, axis=1),
                'cov_deltas': np.cov(deltas),
                'mean_offsets': np.mean(offsets, axis=1),
                'cov_offsets': np.cov(offsets),
            }

    return distribution_parameters


def euler2quat(data, imu_names):
    """Convert intrinsic Euler angles to quaternions for multiple IMUs.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing Euler angle columns named ``"{imu}_Ox"``,
        ``"{imu}_Oy"`` and ``"{imu}_Oz"`` for each IMU.
    imu_names : list of str
        Names of the IMUs that should be converted.

    Returns
    -------
    pandas.DataFrame
        DataFrame with quaternion columns ``"{axis}_{imu}"`` for all axes in
        :data:`QUATERNION_AXES`.
    """
    quats = pd.DataFrame([])
    for imu_name in imu_names:
        euler_cols = [f"{imu_name}_O{axis}" for axis in ["x", "y", "z"]]
        quat_cols = [f"{axis}_{imu_name}" for axis in QUATERNION_AXES]
        rot = R.from_euler(INTRINSIC_EULER_ORDER, np.array(data[euler_cols]), degrees=True)
        quats[quat_cols] = rot.as_quat()
    return quats


def get_imu_name(column_names, imu_pos_name, pos):
    """Resolve the IMU name used for a body position.

    Parameters
    ----------
    column_names : list[str]
        All column names present in the dataset.
    imu_pos_name : dict
        Mapping from body position to IMU name or list of candidates.
    pos : str
        Target body position whose IMU name should be returned.

    Returns
    -------
    str
        Resolved IMU name for the given body position.
    """
    if pos != "head":
        return imu_pos_name[pos]
    else:
        for possible_imu_name in imu_pos_name[pos]:
            if "w_" + possible_imu_name in column_names:
                return possible_imu_name


def find_glitches(quat):
    """Return Euclidean distances between consecutive quaternions.

    Parameters
    ----------
    quat : np.ndarray
        Array of shape ``(T, 4)`` containing a quaternion time series.

    Returns
    -------
    np.ndarray
        Vector of length ``T - 1`` with distances between successive samples.
    """
    distances = np.linalg.norm(quat[:-1] - quat[1:], axis=1)
    return distances


def fix_glitches(quat):
    """Flip quaternion signs to minimize frame-to-frame jumps.

    Parameters
    ----------
    quat : np.ndarray
        Quaternion sequence of shape ``(T, 4)``.

    Returns
    -------
    np.ndarray
        Corrected quaternion sequence with sign flips applied.
    """
    quat = copy.deepcopy(quat)
    for i in range(1, len(quat)):
        q = quat[i, :]
        minus_q = q * -1
        distance_q = np.linalg.norm(quat[i - 1] - q)
        distance_minus_q = np.linalg.norm(quat[i - 1] - minus_q)
        if distance_minus_q < distance_q:
            quat[i, :] = minus_q
    return quat


def unify_quaternions(quat_data, imu_names):
    """Ensure consistent quaternion orientation for multiple IMUs.

    This function checks each quaternion time series for sudden jumps caused by
    sign flips and tries to fix them. Additionally, the sign is unified so that
    the first frame has a positive ``w`` component.

    Parameters
    ----------
    quat_data : pandas.DataFrame
        DataFrame containing quaternion columns for all IMUs.
    imu_names : list[str]
        Names of the IMUs to process.

    Returns
    -------
    pandas.DataFrame
        DataFrame with corrected quaternion columns.
    """
    fixed_data = pd.DataFrame([])
    for imu_name in imu_names:
        col_names = [f"{axes}_{imu_name}" for axes in QUATERNION_AXES]
        quat = quat_data[col_names].to_numpy(copy=True)
        distances = find_glitches(quat)

        if max(distances) > 1.5:
            quat = fix_glitches(quat)

            distances = find_glitches(quat)
            if max(distances) > 1.5:
                print(f"IMU Position {imu_name}")
                raise ValueError("Not all glitches were fixed")

        if quat[0, 3] < 0:
            quat = quat * -1

        fixed_data[col_names] = quat
    return fixed_data


def convert_to_imu(orientation_data, imu_names):
    """Convert Euler angle orientation data to unified quaternion format.

    Parameters
    ----------
    orientation_data : pandas.DataFrame
        DataFrame with Euler angle columns for each IMU.
    imu_names : list[str]
        Names of the IMUs contained in ``orientation_data``.

    Returns
    -------
    pandas.DataFrame
        Quaternion representation with consistent orientation.
    """
    quat_data = euler2quat(orientation_data, imu_names)
    quat_data = unify_quaternions(quat_data, imu_names)
    return quat_data


