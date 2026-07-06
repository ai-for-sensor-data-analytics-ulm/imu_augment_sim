"""Example automatic labelling rules used in :mod:`basic_workflow`."""

from typing import Any
from imu_augment_sim.labeling import Rule


def __convert_string_to_bool(string):
    if not isinstance(string, str):
        return string
    if string.lower() in ['true', '1', 'yes']:
        return True
    else:
        return False


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

    neg_sub = __convert_string_to_bool(results['max_neg_range_subtalar_below']['boolean'])
    pos_sub = __convert_string_to_bool(results['max_pos_range_subtalar_below']['boolean'])
    mtp = __convert_string_to_bool(results['max_range_mtp_below']['boolean'])
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




def rule_max_range_below(data: Any, joint_angle: str, threshold: float) -> dict:
    """Check whether the motion range of ``joint_angle`` exceeds ``threshold``.

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
        Dict with ``'boolean'`` (``True`` if the range of motion is *greater*
        than ``threshold``) and the measured ``max_delta_<joint_angle>``.
    """
    joint_angles = data['joint_angles']
    max_delta = joint_angles[joint_angle].max() - joint_angles[joint_angle].min()
    return {'boolean': max_delta > threshold, f'max_delta_{joint_angle}':max_delta}


def rule_neg_range(data: Any, joint_angle:str, threshold: float) -> dict:
    """Check whether the negative range (start minus minimum) exceeds ``threshold``.

    Parameters
    ----------
    data : Any
        Analysis data containing joint angles.
    joint_angle : str
        Column name of the joint angle to check.
    threshold : float
        Range threshold.

    Returns
    -------
    dict
        Dict with ``'boolean'`` (``True`` if the negative range is *greater*
        than ``threshold``) and the measured ``max_neg_range_<joint_angle>``.
    """
    joint_angles = data['joint_angles']
    neg_range = joint_angles[joint_angle].iloc[0] - min(joint_angles[joint_angle])
    return {'boolean': neg_range > threshold, f'max_neg_range_{joint_angle}': neg_range}

def rule_pos_range(data: Any, joint_angle:str, threshold: float) -> dict:
    """Check whether the positive range (maximum minus start) exceeds ``threshold``.

    Parameters
    ----------
    data : Any
        Analysis data containing joint angles.
    joint_angle : str
        Column name of the joint angle to check.
    threshold : float
        Range threshold.

    Returns
    -------
    dict
        Dict with ``'boolean'`` (``True`` if the positive range is *greater*
        than ``threshold``) and the measured ``max_pos_range_<joint_angle>``.
    """
    joint_angles = data['joint_angles']
    pos_range = max(joint_angles[joint_angle]) - joint_angles[joint_angle].iloc[0]
    return {'boolean': pos_range > threshold, f'max_pos_range_{joint_angle}': pos_range}

al_rules = {
    'max_neg_range_subtalar_below': Rule(
        name='max_neg_range_subtalar_below',
        func=rule_neg_range,
        kwargs={'joint_angle': 'subtalar_angle_r', 'threshold': cfg['thresholds']['neg_subtalar_r']}
    ),
    'max_pos_range_subtalar_below': Rule(
        name='max_pos_range_subtalar_below',
        func=rule_pos_range,
        kwargs={'joint_angle': 'subtalar_angle_r', 'threshold': cfg['thresholds']['pos_subtalar_r']}
    ),
    'max_range_mtp_below': Rule(
        name='max_range_mtp_below',
        func=rule_max_range_below,
        kwargs={'joint_angle': 'mtp_angle_r', 'threshold': cfg['thresholds']['mtp_angle_r']}
    )
}