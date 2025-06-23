"""End‑to‑end augmentation script executed from the repo root.

Usage
-----
python examples/augment_pipeline.py \
       --dataset_root datasets/ \
       --exercise deep_squat \
       --out augmented/ \
       --beta_delta angle_series/beta_delta.npz \
       --ik_model models/rajagopal_scaled.osim
"""
from __future__ import annotations

from src.imu_augment_sim.dataset import ExerciseDataset
from src.imu_augment_sim.preprocessing import calibrate_segments, calculate_distribution_parameters_from_data, convert_to_imu
from src.imu_augment_sim.augmentation import augment_sample
from src.imu_augment_sim.ik import create_opensim_file, perform_ik
from src.imu_augment_sim.labeling import RuleEvaluator
from src.imu_augment_sim.analysis import load_analysis_data, perform_analysis
from pathlib import Path
import json
from util import load_example_data
import yaml
from example_al_rules import al_rules, al_logic


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    """Run the full augmentation and analysis example.

    This script demonstrates the end-to-end pipeline from data loading to
    automatic labelling. It is intended to be executed from the repository
    root.

    Returns
    -------
    None
        The function is executed for its side effects.
    """
    with open('example_config.yml', 'r') as stream:
        cfg = yaml.safe_load(stream)

    # load and prepare angle_series
    example_data, example_data_labels, example_data_unique_ids = load_example_data(Path("../data/example_dataset/CEOD.pkl"))

    # 1. Load dataset
    example_dataset = ExerciseDataset(data=example_data, labels=example_data_labels, unique_ids=example_data_unique_ids,
                                      samplerate=100, augment_indices=[0, 1])
    # calibrate all segments of all repetitions (optional)

    calibrated_data = calibrate_segments(example_dataset.data, example_dataset.available_imu_names, cfg['segment_rotations'])

    # either calculate distribution parameters from data...
    distribution_parameters = calculate_distribution_parameters_from_data(data=calibrated_data,
                                                        imu_names=example_dataset.available_imu_names,
                                                        labels=example_dataset.labels)

    # ...or load them from a file
    # distribution_parameters = pkl.load(open(cfg['paths']['distribution_parameters'], 'rb'))

    for sample, label, uid in zip(example_dataset.data, example_dataset.labels, example_dataset.unique_ids):

        for i, target_label in enumerate(cfg['target_labels']):
            aug_sample = augment_sample(input_sample=sample,
                                        imu_names = example_dataset.available_imu_names,
                                        distribution_parameters=distribution_parameters,
                                        target_label=target_label,
                                        no_scaling=False,
                                        no_shifting=True)

            orientations_file_path = create_opensim_file(filepath=Path(cfg['paths']['ik_output'])  / f'{uid}_{i}',
                                filename='imu_orientations.sto',
                                data=aug_sample,
                                imu_names=example_dataset.available_imu_names,
                                samplerate=example_dataset.samplerate )

            perform_ik(orientations_file_path=orientations_file_path,
                       model_path=Path(cfg['paths']['model']),
                        ik_settings_path=Path(cfg['paths']['ik_settings']),
                        output_path=orientations_file_path.parent)

            perform_analysis(analyzer_settings_path=Path(cfg['paths']['analyzer_settings']),
                             model_path=Path(cfg['paths']['model']),
                             analyzer_results_path=orientations_file_path.parent,
                                coordinates_file_name=orientations_file_path.parent / 'ik_imu_orientations.mot'
                             )

            analysis_data = load_analysis_data(orientations_file_path.parent)

            # perform automatic labeling
            evaluator = RuleEvaluator(analysis_data,al_rules, al_logic)
            label, details = evaluator.evaluate()


            # convert to IMU format and save
            aug_imu_data = convert_to_imu(analysis_data['orientations'], example_dataset.available_imu_names)

            aug_imu_data.to_csv(orientations_file_path.parent / f'augmented_imu_data.csv', index=False)

            with open(orientations_file_path.parent / f'metadata.json', 'w', encoding='utf-8') as f:
                json.dump({'label': label, 'details': details}, f, indent=4, default=str)

if __name__ == "__main__":
    main()