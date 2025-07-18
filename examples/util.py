"""Utility functions for the example scripts."""

from pathlib import Path
import pickle as pkl
from typing import List

def load_example_data(data_path: Path, exercise_identifier:str) -> List[dict]:
    """
    Load example angle_series from the specified path.

    Parameters
    ----------
    data_path : Path
        Path to the example angle_series directory.

    Returns
    -------
    dict
        Dictionary containing the loaded example angle_series.
    """
    datastructure = pkl.load(open(data_path , 'rb'))
    data = [rep['imu_data'] for rep in datastructure['data'] if rep['exercise'] == exercise_identifier]
    labels = [rep['rating'] for rep in datastructure['data']]
    metadata = [{'rep_id': rep['repetition_id'], 'm_id':rep['m_id']} for rep in datastructure['data']]
    return data, labels, metadata
