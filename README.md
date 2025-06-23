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


Unzip it into a folder of your choice. The example workflow will look for the data in `data/example_dataset/`.

## Running the Example

The repository ships with an example workflow illustrating the complete pipeline. Execute:

```bash
python examples/basic_workflow.py
```

The script loads example data, performs augmentation and inverse kinematics and finally analyses the results.

## Customisation

### Switching the Model

The OpenSim model used for inverse kinematics is referenced in `examples/example_config.yml` under `paths.model`. Replace this path with your own `.osim` file to use a different biomechanical model.

### Custom Labelling Rules

Automatic labelling is controlled via rules defined in `examples/example_al_rules.py`. Copy this file and adjust the functions or thresholds to reflect your own assessment criteria. Import your custom module in `basic_workflow.py` to see how different rules influence the final labels.

## Contributing

Bug reports and pull requests are welcome. Please ensure that new code is documented and tested.

## License

This project is released under the MIT license.

![status: under review](https://img.shields.io/badge/status-under--review-yellow)