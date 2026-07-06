> ⚠️ **Repository under construction**
### Publication Status
This repository accompanies the manuscript:

> *Boosting Automatic Exercise Evaluation Through Musculoskeletal Simulation-Based IMU Data Augmentation*  
> Status: Under review (submitted to Nature Scientific Reports)
Preprint: https://arxiv.org/abs/2505.24415

The repository will be finalized and versioned upon acceptance.


# imu-augment-sim

Tools for creating biomechanically realistic IMU data streams using OpenSim 4.4 and for labeling the resulting motion.

## Installation

Clone the repository and install the dependencies. A ready to use conda environment is provided:

```bash
conda env create -f environment.yml
conda activate opensim_scripting_py11
```

Alternatively install the package directly with `pip`:

```bash
pip install -e .
```

Please beware that the opensim package is currently available via conda only. 


## Example Data

To test the augmentation pipeline, download the sample dataset:

Link will be available soon.

Place the pickle file at `data/example_dataset.pkl` (the path configured under
`paths.base_dataset` in the example configs). Adjust that value to point to your own file.

## Data Format

`base_dataset` points to a pickled Python dictionary. Each repetition is a dict; the
relevant fields consumed by `examples/util.py:load_example_data` are:

| Field            | Type                | Description                                                        |
|------------------|---------------------|--------------------------------------------------------------------|
| `exercise`       | `str`               | Exercise identifier (must match `exercise_identifier` in the config). |
| `imu_data`       | `pandas.DataFrame`  | Quaternion columns named `"<axis>_<imu_name>"` with axis in `i, j, k, w`. |
| `rating`         | `int`               | Class label of the repetition.                                     |
| `repetition_id`  | hashable            | Unique repetition identifier.                                      |
| `m_id`           | hashable            | Measurement / subject identifier.                                  |

The dictionary must expose the repetitions under either a `data` or an `X` key. Allowed
IMU names follow `ExerciseDataset.ALLOWED_IMU_NAMES` (e.g. `pelvis`, `femur_r`, `tibia_r`, …).

## Running the Example

The repository ships with an example workflow illustrating the complete pipeline. Execute:

```bash
python examples/basic_workflow.py --config examples/example_config_hs.yml
```

If `--config` is omitted, `examples/example_config_hs.yml` is used by default. One config is
provided per example exercise: `example_config_ds.yml` (deep squat), `example_config_hs.yml`
(hurdle step), `example_config_rd.yml` (resisted dorsiflexion) and `example_config_rgs.yml`
(resisted gait simulation). The script loads example data, performs augmentation and inverse
kinematics and finally analyses the results.

## Customisation

### Switching the Model

The OpenSim model used for inverse kinematics is referenced in the example config
(e.g. `examples/example_config_hs.yml`) under `paths.model`. Replace this path with your own
`.osim` file to use a different biomechanical model.

### Custom Labelling Rules

Automatic labelling is controlled via rule modules named `al_rules_<exercise>.py` in
`examples/` (e.g. `al_rules_hs.py`). The active module is selected through `paths.labeling_fcts`
in the config. Copy one of these files and adjust the functions or thresholds to reflect your own
assessment criteria, then point `paths.labeling_fcts` at your module.

## Contributing

Bug reports and pull requests are welcome. Please ensure that new code is documented and tested.

## License

This project is released under the MIT license. See the [LICENSE](LICENSE) file for the full text.

![status: under review](https://img.shields.io/badge/status-under--review-yellow)