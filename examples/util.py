"""Utility functions for the example scripts."""

from pathlib import Path
import pickle as pkl
from typing import List, Tuple
import json
import pandas as pd

OSIM_SEGMENT_NAMES = {
    'right_foot': 'calcn_r',
    'left_foot': 'calcn_l',
    'right_shin': 'tibia_r',
    'left_shin': 'tibia_l',
    'right_thigh': 'femur_r',
    'left_thigh': 'femur_l',
    'pelvis': 'pelvis',
    'breastbone': 'torso',
    'head': 'head',
    'right_upper_arm': 'humerus_r',
    'left_upper_arm': 'humerus_l',
    'right_forearm': 'radius_r',
    'left_forearm': 'radius_l',
    'right_hand': 'hand_r',
    'left_hand': 'hand_l',
}


def rename_imu_to_osim_segment_names(imu_data, imu_pos_name):
    """Rename IMU quaternion columns to OpenSim body-segment names.

    Parameters
    ----------
    imu_data : pandas.DataFrame
        DataFrame with quaternion columns named ``"<axis>_<imu_name>"``.
    imu_pos_name : dict
        Mapping from body position to IMU name (or list of candidate names).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns renamed to the OpenSim segment names defined in
        :data:`OSIM_SEGMENT_NAMES`. Scapula positions are skipped.
    """
    renamed_angles = pd.DataFrame()
    for pos, imu_name in imu_pos_name.items():
        if pos in ['right_scapula', 'left_scapula']:
            continue
        if type(imu_name) == list:
            for i in imu_name:
                if 'w_' + i  in imu_data.columns:
                    imu_name = i
                    break
        imu_col_names = [el + imu_name for el in ['i_', 'j_', 'k_', 'w_']]
        osim_col_names = [el + OSIM_SEGMENT_NAMES[pos] for el in ['i_', 'j_', 'k_', 'w_']]
        renamed_angles[osim_col_names] = imu_data[imu_col_names]
    return renamed_angles

def load_example_data(data_path: Path, exercise_identifier: str) -> Tuple[List, List, List[dict]]:
    """Load example recordings for a single exercise from a pickle file.

    Parameters
    ----------
    data_path : Path
        Path to the example dataset pickle file.
    exercise_identifier : str
        Only repetitions whose ``exercise`` field matches this identifier are
        returned.

    Returns
    -------
    tuple of (list, list, list[dict])
        A tuple ``(data, labels, metadata)`` where ``data`` is a list of
        per-repetition IMU DataFrames, ``labels`` the corresponding ratings and
        ``metadata`` a list of ``{'rep_id': ..., 'm_id': ...}`` dictionaries.
    """
    datastructure = pkl.load(open(data_path , 'rb'))
    try:
        exercise_data = [rep for rep in datastructure['data'] if rep['exercise'] == exercise_identifier]
    except KeyError:
        exercise_data = [rep for rep in datastructure['X'] if rep['exercise'] == exercise_identifier]
        imu_translation_guide = datastructure['config']['imu_pos_name']
        for rep in exercise_data:
            rep['imu_data'] = rename_imu_to_osim_segment_names(rep['imu_data'], imu_translation_guide)

    labels = [rep['rating'] for rep in exercise_data]
    data = [rep['imu_data'] for rep in exercise_data]
    metadata = [{'rep_id': rep['repetition_id'], 'm_id':rep['m_id']} for rep in exercise_data]
    return data, labels, metadata





def load_progress_file(progress_file):
    """Load the list of already processed rep-ids from a JSON progress file.

    Returns an empty list if the file does not exist yet.
    """
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = list(json.load(f))
    else:
        progress = list()
    return progress


def save_progress_file(progress_file, progress):
    """Persist the list of processed rep-ids to a JSON progress file."""
    with open(progress_file, 'w') as f:
        json.dump(list(progress), f)

