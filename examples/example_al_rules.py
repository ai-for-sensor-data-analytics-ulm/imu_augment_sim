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
    ha = results['max_range_hip_adduction_below']['boolean']
    kf = results['max_range_knee_flexion_below']['boolean']
    ad = results['max_range_ankle_dorsiflexion_below']['boolean']

    if ha and ad:
        return 1
    if ha and not ad:
        return 2
    if not ha and kf:
        return 3
    if not ha and not kf and ad:
        return 1
    if not ha and not kf and not ad:
        return 2


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


al_rules = {
    'max_range_hip_adduction_below': Rule(
        name='max_range_hip_adduction_below',
        func=rule_max_range_below,
        kwargs={'joint_angle': 'hip_adduction_r', 'threshold': cfg['thresholds']['hip_adduction_r']}
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
}