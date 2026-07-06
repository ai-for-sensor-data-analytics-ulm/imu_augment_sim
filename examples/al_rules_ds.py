"""Example automatic labelling rules used in :mod:`basic_workflow`."""

from imu_augment_sim.labeling import Rule
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


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

    fbh_r = results['infos_femur']['max_angle_femur_right_bool']
    fbh_l = results['infos_femur']['max_angle_femur_left_bool']
    fbh = fbh_r and fbh_l

    tpt_r = results['infos_tibia']['parallel_right_bool']
    tpt_l = results['infos_tibia']['parallel_left_bool']
    tpt_t = results['infos_tibia']['torso_vertical_bool']
    tpt = (tpt_r and tpt_l) or tpt_t

    kaf_r = results['infos_knees']['knees_over_toes_right_boolean']
    kaf_l = results['infos_knees']['knees_over_toes_left_boolean']
    kaf = kaf_r and kaf_l

    daf_r = results['infos_dowel']['dowel_over_feet_right_boolean']
    daf_l = results['infos_dowel']['dowel_over_feet_left_boolean']
    daf = daf_r and daf_l

    hog_r = results['infos_heels']['angle_ground_foot_right_boolean']
    hog_l = results['infos_heels']['angle_ground_foot_left_boolean']
    hog = hog_r and hog_l

    nlf = results['infos_lumbar']['angle_pelvis_torso_boolean']


    if fbh and tpt and daf and hog and nlf and kaf:
        rating = 3
    elif fbh and tpt and daf and nlf and kaf and not hog:
        rating = 2
    elif not tpt or not fbh or not nlf or not daf or not kaf:
        rating = 1
    return rating


def calc_deepest_squat_period(joint_df):
    """Return the time window around the deepest point of the squat."""
    lowest_right = joint_df['time'][joint_df['knee_angle_r'].argmax()]
    lowest_left = joint_df['time'][joint_df['knee_angle_l'].argmax()]
    delta_t = np.abs(lowest_left - lowest_right)
    if delta_t < 0.1:
        lowest_right -= 0.1
        lowest_left += 0.1
    return min([lowest_right, lowest_left]), max([lowest_right, lowest_left])


def is_femur_below_horizontal(orientation_array, t_min, t_max, threshold):
    """Check whether the femur drops below the horizontal within a time window."""
    orientation_array = orientation_array[t_min:t_max, :]
    femur_orient = R.from_euler(seq='XYZ', angles=orientation_array, degrees=True)

    # transform orientation into body fixed ZYX representation -> if Z angle is first it can be used to analyze if it is above +90 degrees -> femur is horizontal
    transformed_orient = femur_orient.as_euler('ZYX', degrees=True)
    max_femur_angle = transformed_orient[:, 0].max()
    below_horizontal = max_femur_angle > threshold
    return below_horizontal, max_femur_angle


def find_nearest_index(data, datapoint):
    """Return the index of the first sample whose value exceeds ``datapoint``."""
    return np.where(data > datapoint)[0][0]


def is_femur_below_horizontal_criteria(data, threshold):
    """Evaluate whether both femurs drop below the horizontal during the squat."""
    # find time period of "deepest" squat
    t_start, t_stop = calc_deepest_squat_period(data['joint_angles'])

    right, angle_info_right = is_femur_below_horizontal(orientation_array=np.array(data['orientations'][['femur_r_Ox','femur_r_Oy','femur_r_Oz']]),
                                                        t_min=find_nearest_index(data['orientations']['time'], t_start),
                                                        t_max=find_nearest_index(data['orientations']['time'], t_stop),
                                                        threshold=threshold)

    left, angle_info_left = is_femur_below_horizontal(orientation_array=np.array(data['orientations'][['femur_l_Ox','femur_l_Oy','femur_l_Oz']]),
                                                      t_min=find_nearest_index(data['orientations']['time'], t_start),
                                                      t_max=find_nearest_index(data['orientations']['time'], t_stop),
                                                      threshold=threshold)

    infos = {'max_angle_femur_right': angle_info_right,
             'max_angle_femur_left': angle_info_left,
             'max_angle_femur_right_bool': right,
             'max_angle_femur_left_bool': left}

    return infos


def is_tibia_parallel_to_torso_criteria(data, threshold):
    """Evaluate whether the tibiae stay parallel to the torso during the squat."""
    mot_data = data['joint_angles']
    angle_tibia_vertical_r = mot_data['pelvis_tilt'] * (-1) + 180 - mot_data['hip_flexion_r'] + mot_data['knee_angle_r']
    angle_tibia_vertical_l = mot_data['pelvis_tilt'] * (-1) + 180 - mot_data['hip_flexion_l'] + mot_data['knee_angle_l']
    angle_torso_vertical = mot_data['pelvis_tilt'] * (-1) + 180 - mot_data['lumbar_extension']
    diff_r = np.abs(angle_torso_vertical - angle_tibia_vertical_r)
    diff_l = np.abs(angle_torso_vertical - angle_tibia_vertical_l)

    # combine collected criteria
    # get idx of maxima
    max_right = np.argmax(diff_r)
    max_left = np.argmax(diff_l)
    min_torso = np.argmax(angle_torso_vertical)

    parallel_right = diff_r.iloc[max_right]
    parallel_left = diff_l.iloc[max_left]
    torso_vertical = (angle_torso_vertical.iloc[min_torso] - 180)

    parallel_right_bool = parallel_right < threshold
    parallel_left_bool = parallel_left < threshold
    torso_vertical_bool = torso_vertical < threshold

    infos = {
             'parallel_right': parallel_right,
             'parallel_left': parallel_left,
             'torso_vertical': torso_vertical,
    'parallel_right_bool': parallel_right_bool,
    'parallel_left_bool': parallel_left_bool,
    'torso_vertical_bool': torso_vertical_bool}

    return infos


def is_knee_above_feet(positions_df, threshold, side):
    """Check whether the knee stays above the foot line (toes-ankle) in the x-z plane."""
    if side == 'right':
        ankle_x = np.array(positions_df['ankle_r_x'])
        ankle_z = np.array(positions_df['ankle_r_z'])
        toe_x = np.array(positions_df['feet_r_x'])
        toe_z = np.array(positions_df['feet_r_z'])
        knee_x = np.array(positions_df['knee_r_x'])
        knee_z = np.array(positions_df['knee_r_z'])
    elif side == 'left':
        ankle_x = np.array(positions_df['ankle_l_x'])
        ankle_z = np.array(positions_df['ankle_l_z'])
        toe_x = np.array(positions_df['feet_l_x'])
        toe_z = np.array(positions_df['feet_l_z'])
        knee_x = np.array(positions_df['knee_l_x'])
        knee_z = np.array(positions_df['knee_l_z'])
    else:
        raise ValueError('invalid value for parameter "side"')

    # calculate distance between knee and line (toes - ankle) in x-z plane
    dis = np.abs((toe_x - ankle_x) * (ankle_z-knee_z) - (ankle_x - knee_x) * (toe_z - ankle_z)) / \
          np.sqrt(np.square((toe_x - ankle_x)) + np.square((toe_z - ankle_z)))

    over_threshold = np.where(dis > threshold)

    # check how often value is above threshold
    criteria_fulfilled = over_threshold[0].size < 5

    return criteria_fulfilled, dis


def is_knee_above_feet_criteria(data, threshold):
    """Evaluate the knees-over-toes criterion for both legs."""
    positions_df = data['positions']

    right, max_dis_right = is_knee_above_feet(positions_df, threshold, 'right')
    left, max_dis_left = is_knee_above_feet(positions_df, threshold, 'left')

    infos = {'knees_over_toes_right': max_dis_right,
             'knees_over_toes_right_boolean': right,
             'knees_over_toes_left': max_dis_left,
             'knees_over_toes_left_boolean': left}

    return infos


def is_dowel_above_feet_criteria(data, threshold):
    """Evaluate whether the dowel (wrists) stays above the feet for both sides."""
    positions_df = data['positions']

    difference_right = positions_df['wrist_r_x'] - positions_df['feet_r_x']
    difference_left = positions_df['wrist_l_x'] - positions_df['feet_l_x']


    dowel_over_feet_right = difference_right.max()
    dowel_over_feet_left = difference_left.max()

    right = dowel_over_feet_right < threshold
    left = dowel_over_feet_left < threshold

    infos = {'dowel_over_feet_right': dowel_over_feet_right,
             'dowel_over_feet_right_boolean': right,
             'dowel_over_feet_left': dowel_over_feet_left,
             'dowel_over_feet_left_boolean': left}

    return infos

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_heels_horizontal_axis(orientations_foot):
    """Return the per-frame angle between the foot vector and the x-axis (degrees)."""
    # torso and tibia are represented by a vector [0,1,0] in the global coordinate systems
    foot = [1,0,0]

    # orientation of torso and tibia is loaded with scipy Rotations
    rotations_foot = R.from_euler('XYZ', orientations_foot, degrees=True)

    # apply rotations to torso / tibia vector so the vectors are oriented in global cs as the torso / tibia segments are in open sim
    foot = rotations_foot.apply(foot)

    # project rotated vectors in global cs x-y plane (simply done by removing the z-component, resulting length of projected vector is irrelevant for angle measurements) and calculate angle towards each other
    angles = []
    for idx in range(orientations_foot.shape[0]):
        angles.append(angle_between(foot[idx,:2], [1, 0]) * (180 / math.pi))
    return angles


def heels_on_ground_criteria(data, threshold):
    """Evaluate whether both heels stay on the ground (small foot-to-axis angle)."""
    orientations_df = data['orientations']

    orientations_foot_right = np.array(orientations_df[['calcn_r_Ox', 'calcn_r_Oy', 'calcn_r_Oz']])
    orientations_foot_left = np.array(orientations_df[['calcn_l_Ox', 'calcn_l_Oy', 'calcn_l_Oz']])

    # get angle between feet and x-Axis
    angles_right = angle_heels_horizontal_axis(orientations_foot_right)
    angles_left = angle_heels_horizontal_axis(orientations_foot_left)

    max_angles_right = max(angles_right)
    max_angles_left = max(angles_left)

    infos = {'angle_ground_foot_right': max_angles_right,
             'angle_ground_foot_left': max_angles_left,
             'angle_ground_foot_right_boolean': max_angles_right < threshold,
             'angle_ground_foot_left_boolean': max_angles_left < threshold}

    return infos

def no_lumbar_flexion_criteria(data, threshold):
    """Evaluate whether the absolute lumbar flexion stays below ``threshold``."""
    mot_data = data['joint_angles']
    lumbar_flexion = mot_data['lumbar_extension']

    # the absolute lumbar flexion should not exceed threshold. For synth reasons we also need the actual flexion,
    # therefore the index is computed, so we can access the absolute value for comparison and the actual number for
    # documentation
    idx_max_flexion = np.argmax(np.abs(lumbar_flexion))
    angle_pelvis_torso_boolean = np.abs(lumbar_flexion.iloc[idx_max_flexion]) < threshold

    infos = {'angle_pelvis_torso': lumbar_flexion.iloc[idx_max_flexion],
             'angle_pelvis_torso_boolean': angle_pelvis_torso_boolean}

    return infos


al_rules = {
    'infos_femur': Rule(
        name='infos_femur',
        func=is_femur_below_horizontal_criteria,
        kwargs={'threshold': cfg['thresholds']['t_fbh']}
    ),
    'infos_tibia': Rule(
        name='infos_tibia',
        func=is_tibia_parallel_to_torso_criteria,
        kwargs={'threshold': cfg['thresholds']['t_tpt']}
    ),
    'infos_knees': Rule(
        name='infos_knees',
        func=is_knee_above_feet_criteria,
        kwargs={'threshold': cfg['thresholds']['t_kaf']}
    ),
    'infos_dowel': Rule(
        name='infos_dowel',
        func=is_dowel_above_feet_criteria,
        kwargs={'threshold': cfg['thresholds']['t_daf']}
    ),
    'infos_heels': Rule(
        name='infos_heels',
        func=heels_on_ground_criteria,
        kwargs={'threshold': cfg['thresholds']['t_hog']}
    ),
    'infos_lumbar': Rule(
        name='infos_lumbar',
        func=no_lumbar_flexion_criteria,
        kwargs={'threshold': cfg['thresholds']['t_nlf']}
    ),
}