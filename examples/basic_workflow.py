"""End-to-end augmentation example.

Run from the repository root after installing the package (``pip install -e .``)::

    python examples/basic_workflow.py --config examples/example_config_hs.yml

If ``--config`` is omitted, ``example_config_hs.yml`` next to this script is used.
The script loads example data, performs augmentation and inverse kinematics and
finally analyses the results.
"""
from __future__ import annotations

from imu_augment_sim.dataset import ExerciseDataset
from imu_augment_sim.preprocessing import calibrate_segments, calculate_distribution_parameters_from_data, \
    convert_to_imu
from imu_augment_sim.augmentation import augment_sample
from imu_augment_sim.ik import create_opensim_file, perform_ik, next_folder_name
from imu_augment_sim.labeling import RuleEvaluator
from imu_augment_sim.analysis import load_analysis_data, perform_analysis
from pathlib import Path
import argparse
import importlib.util
import json
from util import load_example_data, load_progress_file, save_progress_file
import yaml


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def get_indices_for_rep_ids(metadata: list, rep_ids: list) -> list:
    """Return the dataset indices whose rep_id is contained in *rep_ids*.

    Parameters
    ----------
    metadata : list[dict]
        Per-repetition metadata as returned by ``load_example_data``.
        Each entry must have a ``'rep_id'`` key.
    rep_ids : list
        Rep-IDs to look up (values must be comparable to the ``'rep_id'``
        values stored in *metadata*).

    Returns
    -------
    list[int]
        Sorted list of indices *i* such that ``metadata[i]['rep_id']`` is
        in *rep_ids*.

    Raises
    ------
    ValueError
        If one or more of the requested *rep_ids* are not found in
        *metadata*.
    """
    rep_id_set = set(rep_ids)
    indices = [i for i, m in enumerate(metadata) if m['rep_id'] in rep_id_set]

    found_ids = {metadata[i]['rep_id'] for i in indices}
    missing = rep_id_set - found_ids
    if missing:
        raise ValueError(f"rep_ids not found in metadata: {sorted(missing)}")

    return sorted(indices)


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to config YAML file. Defaults to example_config_hs.yml next to this script.')
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent

    config_path = args.config if args.config is not None else BASE_DIR / 'example_config_hs.yml'
    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)
        print(f'config loaded {config_path}')

    _rules_path = BASE_DIR / (cfg['paths']['labeling_fcts'] + '.py')
    _spec = importlib.util.spec_from_file_location(cfg['paths']['labeling_fcts'], _rules_path)
    _al_module = importlib.util.module_from_spec(_spec)
    _al_module.cfg = cfg
    _spec.loader.exec_module(_al_module)
    al_rules = _al_module.al_rules
    al_logic = _al_module.al_logic

    progress_file = Path(cfg['paths']['progress_file'])
    progress = load_progress_file(progress_file)

    # load and prepare angle_series
    example_data, example_data_labels, example_data_metadata = load_example_data(
        Path(BASE_DIR.parent / cfg['paths']['base_dataset']), cfg['exercise_identifier'])

    # 1. Load dataset
    rep_ids = cfg.get('rep_ids', None)
    augment_indices = get_indices_for_rep_ids(example_data_metadata, rep_ids) if rep_ids is not None else None

    example_dataset = ExerciseDataset(data=example_data, labels=example_data_labels, metadata=example_data_metadata,
                                      samplerate=100, augment_indices=augment_indices)

    if cfg['calibrate_segments']:
        # calibrate all segments of all repetitions (optional)
        example_dataset.data = calibrate_segments(example_dataset.data, example_dataset.available_imu_names,
                                                  cfg['segment_rotations'])

    if cfg['paths'].get('distribution_parameters') is None:
        distribution_parameters = calculate_distribution_parameters_from_data(data=example_dataset.data,
                                                                              imu_names=example_dataset.available_imu_names,
                                                                              labels=example_dataset.labels,
                                                                              variance_scaling=cfg['variance_scaling'])
    else:
        import pickle as pkl
        with open(cfg['paths']['distribution_parameters'], 'rb') as f:
            distribution_parameters = pkl.load(f)

    generated_samples = 0
    rep_target_labels_map = cfg.get('rep_target_labels', {})

    for idx, (sample, original_label, meta) in enumerate(
            zip(example_dataset.data, example_dataset.labels, example_dataset.metadata)):

        if meta['rep_id'] in progress:
            print(f"Skipping {meta['rep_id']} as it is already processed.")
            continue

        uid = meta['rep_id']  # unique identifier for the repetition
        current_target_labels = rep_target_labels_map.get(uid, cfg['target_labels'])

        for i, target_label in enumerate(current_target_labels):
            folder_name = next_folder_name(base_path=(BASE_DIR / Path(cfg['paths']['ik_output'])).resolve(),
                                           identifier=f'{uid}')

            aug_sample = augment_sample(input_sample=sample,
                                        imu_names=example_dataset.available_imu_names,
                                        distribution_parameters=distribution_parameters,
                                        target_label=target_label,
                                        no_scaling=cfg['no_scaling'],
                                        no_shifting=cfg['no_shifting'])

            orientations_file_path = create_opensim_file(
                filepath=(BASE_DIR / Path(cfg['paths']['ik_output']) / folder_name).resolve(),
                filename='imu_orientations.sto',
                data=aug_sample,
                imu_names=example_dataset.available_imu_names,
                samplerate=example_dataset.samplerate)

            perform_ik(orientations_file_path=orientations_file_path,
                       model_path=(BASE_DIR / Path(cfg['paths']['model'])).resolve(),
                       ik_settings_path=(BASE_DIR / Path(cfg['paths']['ik_settings'])).resolve(),
                       output_path=orientations_file_path.parent)

            perform_analysis(analyzer_settings_path=(BASE_DIR / Path(cfg['paths']['analyzer_settings'])).resolve(),
                             model_path=(BASE_DIR / Path(cfg['paths']['model'])).resolve(),
                             analyzer_results_path=orientations_file_path.parent,
                             coordinates_file_name=orientations_file_path.parent / 'ik_imu_orientations.mot'
                             )

            analysis_data = load_analysis_data(orientations_file_path.parent)

            # perform automatic labeling
            evaluator = RuleEvaluator(analysis_data, al_rules, al_logic)
            label, details = evaluator.evaluate()

            # convert to IMU format and save
            aug_imu_data = convert_to_imu(analysis_data['orientations'], example_dataset.available_imu_names)

            aug_imu_data.to_csv(orientations_file_path.parent / 'augmented_imu_data.csv', index=False)

            with open(orientations_file_path.parent / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump({'label': label, 'details': details, 'target_label': target_label, 'original_rep_id': uid,
                           'original_m_id': meta['m_id'], 'original_label': original_label}, f, indent=4, default=str)

            generated_samples += 1

        progress.append(meta['rep_id'])
        save_progress_file(progress_file, progress)
        if generated_samples > cfg['max_generated_samples']:
            break


if __name__ == "__main__":
    main()
