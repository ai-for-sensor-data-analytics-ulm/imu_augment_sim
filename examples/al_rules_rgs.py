"""Example automatic labelling rules used in :mod:`basic_workflow`."""

from typing import Any
from imu_augment_sim.labeling import Rule



def al_logic(results: dict[str, bool]) -> int:
    """Combine rule results into a single label.

    Parameters
    ----------
    results : dict[str, bool]
        Mapping of rule names to boolean outcomes.

    Returns
    -------
    int
        Label generated from the rule combination logic.
    """
    ha = results['max_range_leg_adduction_below']['boolean']
    kf = results['max_range_knee_flexion_below']['boolean']
    ad = results['min_ankle_dorsiflexion_below']['boolean']
    hf = results['max_range_hip_flexion_below']['boolean']

    if ha:
        if ad:
            rating = 1
        else:
            if hf:
                rating = 2
            else:
                rating = 4
    else:
        if kf:
            rating = 3
        else:
            if ad:
                rating = 1
            else:
                if hf:
                    rating = 2
                else:
                    rating = 4
    return rating


def rule_max_range_below(data: Any, joint_angle: str, threshold: float) -> dict:
    """Check whether the motion range of ``joint_angle`` stays below ``threshold``.

    Parameters
    ----------
    data : Any
        Analysis data containing joint angles.
    joint_angle : str
        Column name of the joint angle to check.
    threshold : float
        Range-of-motion threshold.

    Returns
    -------
    dict
        Dict with ``'boolean'`` (``True`` if the range of motion is *smaller*
        than ``threshold``) and the measured ``max_delta_<joint_angle>``.
    """
    joint_angles = data['joint_angles']
    max_delta = joint_angles[joint_angle].max() - joint_angles[joint_angle].min()
    return {'boolean': max_delta < threshold, f'max_delta_{joint_angle}':max_delta}


def rule_max_combined_range_below(data: Any, joint_angle_0, joint_angle_1, threshold: float) -> dict:
    """Check whether the range of the summed joint angles stays below ``threshold``.

    Parameters
    ----------
    data : Any
        Analysis data containing joint angles.
    joint_angle_0, joint_angle_1 : str
        Column names of the two joint angles that are summed before computing
        the range of motion.
    threshold : float
        Combined range-of-motion threshold.

    Returns
    -------
    dict
        Dict with ``'boolean'`` (``True`` if the combined range of motion is
        *smaller* than ``threshold``) and the measured ``max_delta_combined``.
    """
    joint_angles_data = data['joint_angles']
    combined_joint_angles = joint_angles_data[joint_angle_0] + joint_angles_data[joint_angle_1]
    max_delta = max(combined_joint_angles) - min(combined_joint_angles)
    return {'boolean': max_delta < threshold, 'max_delta_combined': max_delta}


def rule_min_below(data: Any, joint_angle: str, threshold: float) -> dict:
    """Check whether the negated minimum of a joint angle stays below ``threshold``.

    Parameters
    ----------
    data : Any
        Analysis data containing joint angles.
    joint_angle : str
        Column name of the joint angle to check.
    threshold : float
        Threshold for the negated minimum value.

    Returns
    -------
    dict
        Dict with ``'boolean'`` (``True`` if ``-min(joint_angle)`` is *smaller*
        than ``threshold``) and the measured ``min_<joint_angle>``.
    """
    joint_angles = data['joint_angles']
    min_value = min(joint_angles[joint_angle])*-1
    return {'boolean': min_value < threshold, f'min_{joint_angle}': min_value}

al_rules = {
    'max_range_leg_adduction_below': Rule(
        name='max_range_leg_adduction_below',
        func=rule_max_combined_range_below,
        kwargs={'joint_angle_0': 'hip_adduction_r', 'joint_angle_1':'pelvis_list', 'threshold': cfg['thresholds']['leg_adduction_r']}
    ),
    'max_range_knee_flexion_below': Rule(
        name='max_range_knee_flexion_below',
        func=rule_max_range_below,
        kwargs={'joint_angle': 'knee_angle_r', 'threshold': cfg['thresholds']['knee_angle_r']}
    ),
    'min_ankle_dorsiflexion_below': Rule(
        name='min_ankle_dorsiflexion_below',
        func=rule_min_below,
        kwargs={'joint_angle': 'ankle_angle_r', 'threshold': cfg['thresholds']['ankle_angle_r']}
    ),
    'max_range_hip_flexion_below': Rule(
        name='max_range_hip_flexion_below',
        func=rule_max_range_below,
        kwargs={'joint_angle': 'hip_flexion_r', 'threshold': cfg['thresholds']['hip_flexion_r']}
    ),
}