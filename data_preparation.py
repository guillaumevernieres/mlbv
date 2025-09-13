"""
Data preparation utilities for IceNet training.
Replicates the C++ data preparation logic from IceEmul.h in Python.

(C) Copyright 2024 NOAA/NWS/NCEP/EMC
This software is licensed under the terms of the Apache Licence Version 2.0
which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""

import numpy as np
import netCDF4 as nc
import torch
from typing import Tuple, Dict, Optional
from pathlib import Path


def select_data(mask: float, lat: float, aice: float,
                clean_data: bool, pole: str) -> bool:
    """
    Check if data point should be included based on domain criteria.

    Args:
        mask: Ocean mask value (1=ocean, 0=land)
        lat: Latitude in degrees
        aice: Sea ice concentration (0-1)
        clean_data: If True, apply strict quality filters
        pole: Domain pole ("north" or "south")

    Returns:
        True if data point should be included
    """
    if pole == "north":
        if clean_data:
            return (mask == 1 and lat > 40.0 and
                    aice > 0.0 and aice <= 1.0)
        else:
            return lat > 60.0
    elif pole == "south":
        if clean_data:
            return (mask == 1 and lat < -40.0 and
                    aice > 0.0 and aice <= 1.0)
        else:
            return lat < -60.0
    else:
        raise ValueError(f"Invalid pole value: {pole}")


class IceDataPreparer:
    """
    Prepares training data from NetCDF files for IceNet model.
    Replicates the C++ prepData functionality from IceEmul.h.
    """

    def __init__(self, config: Dict):
        """
        Initialize data preparer with configuration.

        Args:
            config: Configuration dictionary with domain settings
        """
        self.config = config
        self.pole = config.get('domain', {}).get('pole', 'north')
        self.clean_data = config.get('domain', {}).get('clean_data', True)

        # Variable names mapping from C++ code
        self.var_names = {
            'lat': 'ULAT',
            'lon': 'ULON',
            'aice': 'aice_h',
            'tsfc': 'Tsfc_h',
            'sst': 'sst_h',
            'sss': 'sss_h',
            'sice': 'sice_h',
            'hi': 'hi_h',
            'hs': 'hs_h',
            'mask': 'umask',
            'tair': 'Tair_h'
        }

    def read_netcdf_data(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Read all required variables from NetCDF file.

        Args:
            filename: Path to NetCDF input file

        Returns:
            Dictionary of variable arrays
        """
        print(f"Reading data from: {filename}")

        with nc.Dataset(filename, 'r') as dataset:
            data = {}

            # Read all variables
            for key, var_name in self.var_names.items():
                if var_name in dataset.variables:
                    data[key] = dataset.variables[var_name][:].flatten()
                    print(f"Read {key} ({var_name}): shape {data[key].shape}")
                else:
                    raise KeyError(
                        f"Variable {var_name} not found in {filename}"
                    )

        return data

    def filter_data(self, data: Dict[str, np.ndarray],
                    max_patterns: int = 400000) -> Tuple[np.ndarray, ...]:
        """
        Filter data based on domain criteria and create training patterns.

        Args:
            data: Dictionary of variable arrays
            max_patterns: Maximum number of patterns to extract

        Returns:
            Tuple of (patterns, targets, lons, lats) as numpy arrays
        """
        print(f"Filtering data for {self.pole} pole, "
              f"clean_data={self.clean_data}")

        # Create selection mask
        n_points = len(data['lat'])
        selected_indices = []

        for i in range(n_points):
            if select_data(float(data['mask'][i]), float(data['lat'][i]), float(data['aice'][i]),
                           self.clean_data, self.pole):
                selected_indices.append(i)
                if len(selected_indices) >= max_patterns:
                    break

        n_patterns = len(selected_indices)
        print(f"Selected {n_patterns} patterns out of {n_points} total points")

        if n_patterns == 0:
            raise ValueError(
                "No valid data points found with current criteria"
            )

        # Create pattern matrix (C++ order: tair, tsfc, sst, sss, hs, hi, sice)
        patterns = np.zeros((n_patterns, 7), dtype=np.float32)
        targets = np.zeros(n_patterns, dtype=np.float32)
        lons = np.zeros(n_patterns, dtype=np.float32)
        lats = np.zeros(n_patterns, dtype=np.float32)

        for cnt, idx in enumerate(selected_indices):
            # Input features (same order as C++ code)
            patterns[cnt, 0] = data['tair'][idx]  # Air temperature
            patterns[cnt, 1] = data['tsfc'][idx]  # Surface temperature
            patterns[cnt, 2] = data['sst'][idx]   # Sea surface temperature
            patterns[cnt, 3] = data['sss'][idx]   # Sea surface salinity
            patterns[cnt, 4] = data['hs'][idx]    # Snow thickness
            patterns[cnt, 5] = data['hi'][idx]    # Ice thickness
            patterns[cnt, 6] = data['sice'][idx]  # Ice salinity

            # Target (ice concentration)
            targets[cnt] = data['aice'][idx]

            # Geolocation
            lons[cnt] = data['lon'][idx]
            lats[cnt] = data['lat'][idx]

        return patterns, targets, lons, lats

    def compute_normalization_stats(
            self, patterns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and standard deviation for input normalization.

        Args:
            patterns: Input patterns array [n_samples, n_features]

        Returns:
            Tuple of (mean, std) arrays
        """
        mean = np.mean(patterns, axis=0).astype(np.float32)
        std = np.std(patterns, axis=0).astype(np.float32)

        # Prevent division by zero (following C++ logic)
        std = np.where(std > 1e-6, std, 1.0)

        print("Normalization statistics:")
        feature_names = ['tair', 'tsfc', 'sst', 'sss', 'hs', 'hi', 'sice']
        for i, name in enumerate(feature_names):
            print(f"  {name}: mean={mean[i]:.3f}, std={std[i]:.3f}")

        return mean, std

    def prepare_training_data(self, filename: str,
                              max_patterns: int = 400000,
                              output_file: Optional[str] = None) -> Dict:
        """
        Complete data preparation pipeline.

        Args:
            filename: Input NetCDF file path
            max_patterns: Maximum number of training patterns
            output_file: Optional output file to save processed data

        Returns:
            Dictionary with processed data
        """
        # Read raw data
        raw_data = self.read_netcdf_data(filename)

        # Filter and create patterns
        patterns, targets, lons, lats = self.filter_data(
            raw_data, max_patterns
        )

        # Compute normalization statistics
        input_mean, input_std = self.compute_normalization_stats(patterns)

        # Create result dictionary
        result = {
            'inputs': patterns,
            'targets': targets,
            'lons': lons,
            'lats': lats,
            'input_mean': input_mean,
            'input_std': input_std,
            'metadata': {
                'pole': self.pole,
                'clean_data': self.clean_data,
                'n_patterns': len(patterns),
                'input_features': [
                    'tair', 'tsfc', 'sst', 'sss', 'hs', 'hi', 'sice'
                ],
                'target': 'aice'
            }
        }

        # Save to file if requested
        if output_file:
            self.save_processed_data(result, output_file)

        return result

    def save_processed_data(self, data: Dict, filename: str) -> None:
        """
        Save processed data to file.

        Args:
            data: Processed data dictionary
            filename: Output filename (.npz or .pt)
        """
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filename.endswith('.npz'):
            # Save as numpy format
            np.savez_compressed(
                filename,
                inputs=data['inputs'],
                targets=data['targets'],
                lons=data['lons'],
                lats=data['lats'],
                input_mean=data['input_mean'],
                input_std=data['input_std'],
                metadata=data['metadata']
            )
        elif filename.endswith('.pt'):
            # Save as PyTorch format
            torch_data = {
                'inputs': torch.from_numpy(data['inputs']),
                'targets': torch.from_numpy(data['targets']),
                'lons': torch.from_numpy(data['lons']),
                'lats': torch.from_numpy(data['lats']),
                'input_mean': torch.from_numpy(data['input_mean']),
                'input_std': torch.from_numpy(data['input_std']),
                'metadata': data['metadata']
            }
            torch.save(torch_data, filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

        print(f"Saved processed data to: {filename}")
        print(f"Data shape: inputs={data['inputs'].shape}, "
              f"targets={data['targets'].shape}")


def create_training_data_from_netcdf(netcdf_file: str,
                                     config: Dict,
                                     output_file: str,
                                     max_patterns: int = 400000) -> str:
    """
    Convenience function to create training data from NetCDF file.

    Args:
        netcdf_file: Input NetCDF file with ocean/ice data
        config: Configuration with domain settings
        output_file: Output file for processed training data
        max_patterns: Maximum number of training patterns

    Returns:
        Path to saved training data file
    """
    preparer = IceDataPreparer(config)
    preparer.prepare_training_data(
        netcdf_file,
        max_patterns=max_patterns,
        output_file=output_file
    )
    return output_file


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare IceNet training data from NetCDF'
    )
    parser.add_argument('input_file', help='Input NetCDF file')
    parser.add_argument('output_file',
                        help='Output training data file (.npz or .pt)')
    parser.add_argument('--pole', choices=['north', 'south'], default='north',
                        help='Polar domain')
    parser.add_argument('--clean-data', action='store_true', default=True,
                        help='Apply strict data quality filters')
    parser.add_argument('--max-patterns', type=int, default=400000,
                        help='Maximum number of training patterns')

    args = parser.parse_args()

    config = {
        'domain': {
            'pole': args.pole,
            'clean_data': args.clean_data
        }
    }

    create_training_data_from_netcdf(
        args.input_file,
        config,
        args.output_file,
        args.max_patterns
    )
