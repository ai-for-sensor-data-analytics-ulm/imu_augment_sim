"""Lightweight container class for IMU recordings."""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

__all__ = [
    "ExerciseDataset",
]



class ExerciseDataset:
    """Generic dataset wrapper for IMU recordings of one exercise.

    Parameters
    ----------
    data : list[pandas.DataFrame]
        Each DataFrame contains quaternion columns for one repetition.
    labels : list[int]
        Numeric label for each repetition.
    metadata : list[dict]
        Per-repetition metadata dictionary.
    samplerate : int
        Sampling frequency of the recordings.
    augment_indices : list[int], optional
        Subset indices that should be augmented.
    distribution_idx : list[int], optional
        Indices used when computing augmentation statistics.
    """
    ALLOWED_IMU_NAMES = ['pelvis', 'femur_r', 'tibia_r', 'calcn_r', 'toes_r',
                         'femur_l', 'tibia_l', 'calcn_l', 'toes_l', 'torso', 'head',
                         'humerus_r', 'radius_r', 'hand_r', 'humerus_l', 'radius_l',
                         'hand_l']
    def __init__(
        self,
        data: List[pd.DataFrame],
        labels: List[int],
        metadata: List[Dict[str, Any]],
        samplerate: int,
        augment_indices: List[int]=None,
        distribution_idx: List[int] = None,
        ) -> None:
        """Instantiate the dataset object.

        Parameters
        ----------
        data : list[pandas.DataFrame]
            IMU data for each repetition.
        labels : list[int]
            Numeric label for each repetition.
        metadata : list[dict]
            Per-repetition metadata dictionary.
        samplerate : int
            Sampling frequency of the recordings.
        augment_indices : list[int], optional
            Subset indices that should be augmented.
        distribution_idx : list[int], optional
            Indices used when computing augmentation statistics.
        """
        if augment_indices is not None:
            data = [data[i] for i in augment_indices]
            labels = [labels[i] for i in augment_indices]
            metadata = [metadata[i] for i in augment_indices]

        self.data = data
        self.metadata = metadata
        self.samplerate = samplerate
        self.augment_idx = list(range(len(data)))
        self.dist_idx = distribution_idx if distribution_idx is not None else list(range(len(data)))
        self.labels = labels

        # Column convention is already validated by the ``data`` setter above.
        self._validate_indices()
        self.available_imu_names = self._extract_imu_names()

        return

        # ------------------------------------------------------------------

    def _validate_column_convention(self) -> None:
        """Validate IMU column names of all samples.

        Raises
        ------
        ValueError
            If any sample contains columns that do not follow the
            ``"<axis>_<imu_name>"`` naming convention.
        """
        import re

        AXES_PATTERN = r"[ijkw]"
        POSITIONS_PATTERN = "|".join(map(re.escape, self.ALLOWED_IMU_NAMES)) or r"[a-zA-Z0-9]+"
        NAMING_PATTERN_DEFAULT = rf"^(?:{AXES_PATTERN})_(?:{POSITIONS_PATTERN})$"

        pat = re.compile(NAMING_PATTERN_DEFAULT)
        for i, df in enumerate(self.data):
            bad_cols = [c for c in df.columns if not pat.match(c)]
            if bad_cols:
                raise ValueError(
                    f"Sample {i} violates naming convention – offending columns: {bad_cols}"
                )

        # ------------------------------------------------------------------

    def _validate_indices(self) -> None:
        """Ensure index lists do not contain out-of-range values.

        Raises
        ------
        IndexError
            If ``distribution_idx`` or ``augment_idx`` contains indices
            outside the valid range ``0..len(data) - 1``.
        """
        n = len(self.data)
        for name, idx_set in (
            ("distribution_indices", self.dist_idx),
            ("augment_indices", self.augment_idx),
        ):
            if any(j >= n or j < 0 for j in idx_set):
                raise IndexError(f"{name} contains out-of-range indices (0…{n - 1})")

    def _extract_imu_names(self):
        """Return the set of IMU names present in the dataset.

        Returns
        -------
        list[str]
            Unique IMU names derived from the dataframe columns.
        """
        col_names = self.data[0].columns
        return list(set([col[2:] for col in col_names]))


        # ------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D401
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Count of stored samples.
        """
        return len(self.data)


    @property
    def data(self) -> List[pd.DataFrame]:
        """Return the stored IMU data."""
        return self._data


    @data.setter
    def data(self, value: List[pd.DataFrame]) -> None:
        """Set the IMU data and validate the column naming convention."""
        self._data = value
        self._validate_column_convention()

