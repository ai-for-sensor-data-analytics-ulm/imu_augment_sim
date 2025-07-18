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

    neg_sub = results['max_neg_range_subtalar_below']['boolean']
    pos_sub = results['max_pos_range_subtalar_below']['boolean']
    mtp = results['max_range_mtp_below']['boolean']
    neg_sub_value = results['max_neg_range_subtalar_below']['max_neg_range_subtalar_angle_r']
    pos_sub_value = results['max_pos_range_subtalar_below']['max_pos_range_subtalar_angle_r']

    evals = (bool(mtp), bool(neg_sub), bool(pos_sub))

    match evals:
        case (False, False, False):
            rating = 1
        case (False, False, True):
            rating = 3
        case (False, True, False):
            rating = 4
        case (False, True, True):
            if neg_sub_value >= pos_sub_value:
                rating = 4
            else:
                rating = 3
        case (True, False, False):
            rating = 2
        case (True, False, True):
            rating = 3
        case (True, True, False):
            rating = 4
        case (True, True, True):
            if neg_sub_value >= pos_sub_value:
                rating = 4
            else:
                rating = 3
        case _:
            raise ValueError("Unexpected evaluation result")
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


def rule_neg_range(data: Any, joint_angle:str, threshold: float) -> bool:
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
    joint_angles = data['joint_angles']
    neg_range = joint_angles[joint_angle].iloc[0] - min(joint_angles[joint_angle])
    return {'boolean': neg_range < threshold, f'max_neg_range_{joint_angle}': neg_range}

def rule_pos_range(data: Any, joint_angle:str, threshold: float) -> bool:
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
    joint_angles = data['joint_angles']
    pos_range = max(joint_angles[joint_angle]) - joint_angles[joint_angle].iloc[0]
    return {'boolean': pos_range < threshold, f'max_pos_range_{joint_angle}': pos_range}

al_rules = {
    'max_neg_range_subtalar_below': Rule(
        name='max_neg_range_subtalar_below',
        func=rule_neg_range,
        kwargs={'joint_angle': 'subtalar_angle_r', 'threshold': cfg['thresholds']['pos_subtalar_r']}
    ),
    'max_pos_range_subtalar_below': Rule(
        name='max_pos_range_subtalar_below',
        func=rule_pos_range,
        kwargs={'joint_angle': 'subtalar_angle_r', 'threshold': cfg['thresholds']['neg_subtalar_r']}
    ),
    'max_range_mtp_below': Rule(
        name='max_range_mtp_below',
        func=rule_max_range_below,
        kwargs={'joint_angle': 'mtp_angle_r', 'threshold': cfg['thresholds']['mtp_angle_r']}
    )
}