"""Utility functions for the example scripts."""

from pathlib import Path
import pickle as pkl
from typing import List

def load_example_data(data_path: Path) -> List[dict]:
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
    example_data = [rep['imu_data'] for rep in datastructure['X']]
    example_data_labels = [rep['rating'] for rep in datastructure['X']]
    example_data_unique_ids = [rep['repetition_id'] for rep in datastructure['X']]
    return example_data, example_data_labels, example_data_unique_ids
