"""Example automatic labelling rules used in :mod:`basic_workflow`."""

from imu_augment_sim.labeling import Rule
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import pandas as pd


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

    hka = results['infos_hka']['max_y_boolean']
    dh = results['infos_dowel_hurdle']['max_angle_dowel_hurdle_boolean']
    lme = results['infos_lumbar']['max_angle_lumbar_extension_boolean']
    lmb = results['infos_lumbar']['max_angle_lumbar_bending_boolean']
    lmr = results['infos_lumbar']['max_angle_lumbar_rotation_boolean']
    nmls = lme and lmb and lmr

    ba = results['infos_balance']['angle_boolean']
    bv = results['infos_balance']['velocity_boolean']
    dist_r = results['infos_balance']['max_distance_r_boolean']
    dist_l = results['infos_balance']['max_distance_l_boolean']

    bal = (ba or bv) and (dist_r or dist_l)

    if hka and dh and nmls and bal:
        rating = 3
    elif (
            not hka or not dh or not nmls) and bal:
        rating = 2
    elif not bal:
        rating = 1
    return rating


def are_hips_knees_ankles_sagittal(data, threshold, side='left'):
    """Evaluate whether hip, knee and ankle stay in the sagittal plane for one leg."""
    orientations_df = data['orientations']
    femur = [0, -1, 0]
    tibia = [0, -1, 0]

    if side == 'right':
        side = 'r'
    elif side == 'left':
        side = 'l'
    else:
        raise ValueError('invalid value for parameter "side"')

    rotations_femur = R.from_euler('XYZ', np.array(
        orientations_df[['femur_' + side + '_Ox', 'femur_' + side + '_Oy', 'femur_' + side + '_Oz']]), degrees=True)
    rotations_tibia = R.from_euler('XYZ', np.array(
        orientations_df[['tibia_' + side + '_Ox', 'tibia_' + side + '_Oy', 'tibia_' + side + '_Oz']]), degrees=True)
    rotations_pelvis = R.from_euler('XYZ', np.array(orientations_df[['pelvis_Ox', 'pelvis_Oy', 'pelvis_Oz']]),
                                    degrees=True)
    femur = rotations_femur.apply(femur)
    tibia = rotations_tibia.apply(tibia)
    norm_vector = np.cross(femur, tibia)

    max_y = max(abs(norm_vector[:,1]))

    criteria = max_y < threshold

    infos = {'max_y': max_y,
             'max_y_boolean': criteria}

    return infos


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def is_dowel_parallel_to_hurdle(data, threshold):
    """Evaluate whether the dowel (defined by the wrists) stays parallel to the hurdle."""
    positions_df = data['positions']
    hurdle = np.array([0, 0, 1])
    # define dowel vector based on the hand positions
    wrist_r = np.array([positions_df['wrist_r_x'], positions_df['wrist_r_y'], positions_df['wrist_r_z']]).T
    wrist_l = np.array([positions_df['wrist_l_x'], positions_df['wrist_l_y'], positions_df['wrist_l_z']]).T
    dowel = wrist_r - wrist_l

    dowel_hurdle_angles = []
    for idx in range(positions_df.shape[0]):
        dowel_angle = (angle_between(hurdle, dowel[idx, :]) * (180 / math.pi))
        dowel_hurdle_angles.append(dowel_angle)

    max_dowel_hurdle = max(abs(np.array(dowel_hurdle_angles)))

    infos = {'max_angle_dowel_hurdle': max_dowel_hurdle,
             'max_angle_dowel_hurdle_boolean': max_dowel_hurdle < threshold}

    return infos


def is_no_movement_lumbar_spine(data, threshold_ext, threshold_bend, threshold_rot):
    """Evaluate whether lumbar extension, bending and rotation ranges stay below thresholds."""
    mot_data = data['joint_angles']
    max_ext = max(mot_data['lumbar_extension']) - min(mot_data['lumbar_extension'])
    max_bend = max(mot_data['lumbar_bending']) - min(mot_data['lumbar_bending'])
    max_rot = max(mot_data['lumbar_rotation']) - min(mot_data['lumbar_rotation'])

    infos = {'max_angle_lumbar_extension': max_ext,
             'max_angle_lumbar_bending': max_bend,
             'max_angle_lumbar_rotation': max_rot,
             'max_angle_lumbar_extension_boolean': max_ext < threshold_ext,
             'max_angle_lumbar_bending_boolean': max_bend < threshold_bend,
             'max_angle_lumbar_rotation_boolean': max_rot < threshold_rot}
    return infos


def calculate_distance(row, pos_zero):
    """Return the Euclidean distance between ``row`` and the reference ``pos_zero``."""
    v = row - pos_zero

    return np.linalg.norm(v.values)


def evaluate_balance(pos_data, col_names):
    """Return the maximum displacement of a hand from its initial position."""
    # get position data of hand
    hand_pos_data = pd.DataFrame()
    for c in col_names:
        hand_pos_data[c] = pos_data[c]
    pos_zero = hand_pos_data.iloc[0]
    hand_pos_data['distance'] = hand_pos_data.apply(calculate_distance, axis=1, args=(pos_zero,))
    max_distance = hand_pos_data['distance'].max()
    return max_distance


def is_balance_maintained(data, threshold_angle, threshold_velocity, threshold_distance):
    """Evaluate the balance criterion from trunk angle/velocity ranges and hand displacement."""
    mot_data = data['joint_angles']
    pos_data = data['positions']
    comb_ext = mot_data['lumbar_extension'] + mot_data['pelvis_tilt']
    max_ext = max(comb_ext) - min(comb_ext)
    max_ext_v = max(abs(np.diff(comb_ext, n=1)))
    comb_bend = mot_data['lumbar_bending'] + mot_data['pelvis_list']
    max_bend = max(comb_bend) - min(comb_bend)
    max_bend_v = max(abs(np.diff(comb_bend, n=1)))
    comb_rot = mot_data['lumbar_rotation'] + mot_data['pelvis_rotation']
    max_rot = max(comb_rot) - min(comb_rot)
    max_rot_v = max(abs(np.diff(comb_rot, n=1)))

    ba = max(max_ext, max_bend, max_rot) < threshold_angle
    bv = max(max_ext_v, max_bend_v, max_rot_v ) < threshold_velocity

    dist_r = evaluate_balance(pos_data, ['hand_r_x', 'hand_r_y', 'hand_r_z'])
    dist_l = evaluate_balance(pos_data, ['hand_l_x', 'hand_l_y', 'hand_l_z'])


    infos = {'max_angle_extension': max_ext,
             'max_angle_bending': max_bend,
             'max_angle_rotation': max_rot,
             'angle_boolean': ba,
             'max_velocity_extension': max_ext_v,
             'max_velocity_bending': max_bend_v,
             'max_velocity_rotation': max_rot_v,
             'velocity_boolean': bv,
             'max_distance_r': dist_r,
             'max_distance_r_boolean': dist_r < threshold_distance,
             'max_distance_l': dist_l,
             'max_distance_l_boolean': dist_l < threshold_distance,}

    return infos


al_rules = {
    'infos_hka': Rule(
        name='infos_hka',
        func=are_hips_knees_ankles_sagittal,
        kwargs={'threshold': cfg['thresholds']['t_hka']}
    ),
    'infos_dowel_hurdle': Rule(
        name='infos_dowel_hurdle',
        func=is_dowel_parallel_to_hurdle,
        kwargs={'threshold': cfg['thresholds']['t_dh']}
    ),
    'infos_lumbar': Rule(
        name='infos_lumbar',
        func=is_no_movement_lumbar_spine,
        kwargs={'threshold_ext': cfg['thresholds']['t_lme'],
                'threshold_bend': cfg['thresholds']['t_lmb'],
                'threshold_rot': cfg['thresholds']['t_lmr']}
    ),
    'infos_balance': Rule(
        name='infos_balance',
        func=is_balance_maintained,
        kwargs={'threshold_angle': cfg['thresholds']['t_ba'],
                'threshold_velocity': cfg['thresholds']['t_bv'],
                'threshold_distance': cfg['thresholds']['t_dist']}
    ),
}