"""Example automatic labelling rules used in :mod:`basic_workflow`."""

from typing import Any, Callable
from src.imu_augment_sim.labeling import Rule
import yaml

with open('example_config.yml', 'r') as stream:
    cfg = yaml.safe_load(stream)


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
    ad = results['max_range_ankle_dorsiflexion_below']['boolean']
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


def rule_max_range_below(data: Any, joint_angle: str, threshold: float) -> bool:
    """Check if the motion range of ``joint_angle`` stays below ``threshold``.

    Parameters
    ----------
    data : Any
        Analysis data containing joint angles.
    joint_angle : str
        Column name of the joint angle to check.
    threshold : float
        Maximum allowed range of motion.

    Returns
    -------
    bool
        ``True`` if the range of motion is smaller than ``threshold``.
    """
    joint_angles = data['joint_angles']
    max_delta = joint_angles[joint_angle].max() - joint_angles[joint_angle].min()
    return {'boolean': max_delta < threshold, f'max_delta_{joint_angle}':max_delta}


def rule_max_combined_range_below(data: Any, joint_angle_0, joint_angle_1, threshold: float) -> bool:
    """Check if the combined range of multiple joint angles stays below a threshold.

    Parameters
    ----------
    data : Any
        Analysis data containing joint angles.
    joint_angles : list[str]
        List of joint angle column names to check.
    threshold : float
        Maximum allowed combined range of motion.

    Returns
    -------
    bool
        ``True`` if the combined range of motion is smaller than ``threshold``.
    """
    joint_angles_data = data['joint_angles']
    combined_joint_angles =  joint_angles_data[joint_angle_0] + joint_angles_data[joint_angle_1]
    max_delta = max(combined_joint_angles) - min(combined_joint_angles)
    return {'boolean': max_delta < threshold, 'max_delta_combined': max_delta}

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
    'max_range_ankle_dorsiflexion_below': Rule(
        name='max_range_ankle_dorsiflexion_below',
        func=rule_max_range_below,
        kwargs={'joint_angle': 'ankle_angle_r', 'threshold': cfg['thresholds']['ankle_angle_r']}
    ),
    'max_range_hip_flexion_below': Rule(
        name='max_range_hip_flexion_below',
        func=rule_max_range_below,
        kwargs={'joint_angle': 'hip_flexion_r', 'threshold': cfg['thresholds']['hip_flexion_r']}
    ),
}