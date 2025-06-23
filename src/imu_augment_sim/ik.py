"""Utilities for writing OpenSim input files and running IK."""

import os
import pandas as pd
import numpy as np
import opensim as osim
import xml.etree.ElementTree as ET


def create_opensim_file(filepath, filename, data, imu_names, samplerate):
    """Write quaternion data to an OpenSim ``.sto`` file.

    Parameters
    ----------
    filepath : Path
        Output directory.
    filename : str
        Name of the ``.sto`` file.
    data : pandas.DataFrame
        DataFrame containing quaternion columns.
    imu_names : list[str]
        Names of IMUs present in ``data``.
    samplerate : int
        Sampling frequency of the data.

    Returns
    -------
    Path
        Path to the written ``.sto`` file.
    """

    str_data = pd.DataFrame()
    time_increment = 1/samplerate
    str_data['time'] = np.arange(data.shape[0]/samplerate, step=time_increment)
    str_data['time'] = str_data['time'].astype(str)
    for imu in imu_names:
        quat_cols = [f"{axis}_{imu}" for axis in ['w', 'i', 'j', 'k']]
        str_data[imu] = data[quat_cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1).to_list()

    header = [f'DataRate={samplerate}', '\nDataType=Quaternion', '\nversion=3',
              '\nOpenSimVersion=4.3-2021-08-27-4bc7ad9', '\nendheader\n']

    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)

    with open(filepath / filename, 'w') as f:
        f.writelines(header)

        f.write('\t'.join(list(str_data.columns)) + '\n')

        for row in str_data.values.tolist():
            row_str = '\t'.join(row) + '\n'
            f.write(row_str)
    return filepath / filename


def perform_ik(orientations_file_path, model_path, ik_settings_path, output_path):
    """Run the OpenSim IMU inverse kinematics tool.

    Parameters
    ----------
    orientations_file_path : Path
        File with IMU orientations in OpenSim format.
    model_path : Path
        Path to the OpenSim model.
    ik_settings_path : Path
        XML settings file controlling the IK tool.
    output_path : Path
        Directory where results are written.

    Returns
    -------
    None
        ``None`` is returned after the tool has finished.
    """

    ik_tool = osim.IMUInverseKinematicsTool(str(ik_settings_path))
    ik_tool.set_model_file(str(model_path))
    ik_tool.set_orientations_file(str(orientations_file_path))
    ik_tool.set_results_directory(str(output_path))
    ik_tool.run(False)

    return


