"""
Data preparation utilities for IceNet training.
Replicates the C++ data preparation logic from IceEmul.h in Python.
"""

import numpy as np
import netCDF4 as nc
import torch
from typing import Tuple, Dict, Optional
from pathlib import Path


def select_data(mask: float, lat: float, aice: float, sst: float,
                clean_data: bool, pole: str, min_ice: float = 0.0) -> bool:
    """
    Check if data point should be included based on domain criteria.

    Args:
        mask: Ocean mask value (1=ocean, 0=land)
        lat: Latitude in degrees
        aice: Sea ice concentration (0-1)
        sst: Sea surface temperature in Celsius
        clean_data: If True, apply strict quality filters
        pole: Domain pole ("north", "south", or "both")
        min_ice: Minimum ice concentration threshold (default: 0.0)

    Returns:
        True if data point should be included
    """
    # Always filter out warm water (SST > 5°C)
    if sst > 5.0:
        return False

    # Apply minimum ice concentration filter
    if aice < min_ice:
        return False

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
    elif pole == "both":
        if clean_data:
            return (mask == 1 and (lat > 40.0 or lat < -40.0) and
                    aice >= 0.0 and aice <= 1.0)
        else:
            return lat > 60.0 or lat < -60.0
    else:
        raise ValueError(f"Invalid pole value: {pole}. "
                         f"Use 'north', 'south', or 'both'.")


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
        self.min_ice = config.get('domain', {}).get('min_ice_concentration', 0.0)
        # Add synthetic data option for testing
        self.use_synthetic = config.get('domain', {}).get('use_synthetic_data', False)

        # Get input size from model config
        self.input_size = config.get('model', {}).get('input_size', 1)

        # Define available input features in order of preference
        self.available_features = [
            'sst',    # Sea surface temperature
            'sss',    # Sea surface salinity
            'tair',   # Air temperature
            'tsfc',   # Surface temperature
            'hi',     # Ice thickness
            'hs',     # Snow thickness
            'sice',   # Ice salinity
            # Stress variables (ice temp vars removed due to NaN issues)
            'strocnx',    # Ocean/ice stress (x)
            'strocny',    # Ocean/ice stress (y)
            'strairx',    # Atm/ice stress (x)
            'strairy',    # Atm/ice stress (y)
            # Heat flux variables
            'qref',       # 2m reference specific humidity
            'flwdn',      # Down longwave flux
            'fswdn',      # Down solar flux
        ]

        # Select features based on input_size
        self.selected_features = self.available_features[:self.input_size]
        print(f"Using {self.input_size} input features: {self.selected_features}")

        # Variable names mapping from C++ code with alternatives
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
            'tair': 'Tair_h',
            'frzmlt': 'frzmlt_h',
            # Ice temperature variables
            #'sitempbot': 'sitempbot_h',
            #'sitempsnic': 'sitempsnic_h',
            #'sitemptop': 'sitemptop_h',
            # Stress variables
            'strocnx': 'strocnx_h',
            'strocny': 'strocny_h',
            'strairx': 'strairx_h',
            'strairy': 'strairy_h',
            # Heat flux variables
            #'fhocn': 'fhocn_h',
            'qref': 'Qref_h',
            #'flwup': 'flwup_h',
            #'fsens': 'fsens_h',
            #'flat': 'flat_h',
            'flwdn': 'flwdn_h',
            'fswdn': 'fswdn_h'
        }

        # Alternative variable name mappings for different file formats
        self.alt_var_names = {
            'aice_h': ['aicen', 'aice', 'ice_concentration'],
            'hi_h': ['hicen', 'hi', 'ice_thickness'],
            'hs_h': ['hsnon', 'hs', 'snow_thickness'],
            'ULAT': ['lat', 'latitude', 'TLAT'],
            'ULON': ['lon', 'longitude', 'TLON'],
            'umask': ['mask', 'land_mask', 'ocean_mask'],
            'Tair_h': ['tair', 'air_temperature'],
            'Tsfc_h': ['tsfc', 'surface_temperature'],
            'sst_h': ['sst', 'sea_surface_temperature'],
            'sss_h': ['sss', 'sea_surface_salinity'],
            'sice_h': ['sice', 'ice_salinity'],
            'frzmlt_h': ['frzmlt', 'frazil_melt', 'frazil_ice_melt']
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

            # Read all variables with fallback to alternative names
            for key, var_name in self.var_names.items():
                found_var = None

                # First try the primary variable name
                if var_name in dataset.variables:
                    found_var = var_name
                else:
                    # Try alternative names
                    alt_names = self.alt_var_names.get(var_name, [])
                    for alt_name in alt_names:
                        if alt_name in dataset.variables:
                            found_var = alt_name
                            print(f"Alternative: {var_name} -> {alt_name}")
                            break

                if found_var:
                    data[key] = dataset.variables[found_var][:].flatten()
                    print(f"Read {key} ({found_var}): shape {data[key].shape}")
                else:
                    available_vars = ', '.join(
                        sorted(dataset.variables.keys())[:10]
                    )
                    raise KeyError(
                        f"Variable {var_name} (or alternatives {alt_names}) "
                        f"not found in {filename}. "
                        f"Available variables (first 10): {available_vars}..."
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
            if select_data(
                float(data['mask'][i]),
                float(data['lat'][i]),
                float(data['aice'][i]),
                float(data['sst'][i]),
                self.clean_data,
                self.pole,
                self.min_ice
            ):
                selected_indices.append(i)
                if len(selected_indices) >= max_patterns:
                    break

        n_patterns = len(selected_indices)
        print(f"Selected {n_patterns} patterns out of {n_points} total points")

        if n_patterns == 0:
            raise ValueError(
                "No valid data points found with current criteria"
            )

        # Create pattern matrix with dynamic number of features
        patterns = np.zeros((n_patterns, self.input_size), dtype=np.float32)
        targets = np.zeros(n_patterns, dtype=np.float32)
        lons = np.zeros(n_patterns, dtype=np.float32)
        lats = np.zeros(n_patterns, dtype=np.float32)

        for cnt, idx in enumerate(selected_indices):
            # Fill input features dynamically based on selected_features
            for i, feature_name in enumerate(self.selected_features):
                if feature_name in data:
                    patterns[cnt, i] = data[feature_name][idx]
                else:
                    # Fill with default value if feature is missing
                    #print(f"Warning: Feature {feature_name} not found, using 0.0")
                    patterns[cnt, i] = 0.0

            # Target (ice concentration)
            targets[cnt] = data['aice'][idx]

            # Apply synthetic data transformation if enabled
            if self.use_synthetic:
                # Simple rule: if SST < -1.4°C, set ice concentration to 0.9
                sst_val = patterns[cnt, 0]  # Assume first feature is SST
                if sst_val < -1.4:
                    targets[cnt] = 0.9
                else:
                    targets[cnt] = 0.0  # Open water for warmer SST

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
        for i, name in enumerate(self.selected_features):
            if i < len(mean):
                print(f"  {name}: mean={mean[i]:.3f}, std={std[i]:.3f}")

        return mean, std

    def thin_patterns(self, patterns, targets, lons, lats, fraction):
        """
        Randomly thin the dataset to a given fraction.
        """
        if fraction < 1.0:
            n = len(targets)
            n_thin = int(n * fraction)
            idx = np.random.choice(n, n_thin, replace=False)
            return patterns[idx], targets[idx], lons[idx], lats[idx]
        return patterns, targets, lons, lats

    def prepare_training_data(self, filename: str,
                              max_patterns: int = 400000,
                              output_file: Optional[str] = None,
                              thin_fraction: float = 1.0) -> Dict:
        """
        Complete data preparation pipeline.

        Args:
            filename: Input NetCDF file path
            max_patterns: Maximum number of training patterns
            output_file: Optional output file to save processed data
            thin_fraction: Fraction of data to keep (default 1.0, no thinning)

        Returns:
            Dictionary with processed data
        """
        # Read raw data
        raw_data = self.read_netcdf_data(filename)

        # Filter and create patterns
        patterns, targets, lons, lats = self.filter_data(
            raw_data, max_patterns
        )

        # Thin the dataset if requested
        patterns, targets, lons, lats = self.thin_patterns(
            patterns, targets, lons, lats, thin_fraction)

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
                'input_features': self.selected_features,
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
