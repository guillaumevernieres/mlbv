#!/usr/bin/env python3
"""
Example script demonstrating how to use the IceNet Python training system
with NetCDF data preparation.

Usage examples:
  # Create and train with synthetic data
  python example_usage.py --create-sample-data

  # Train with the real CICE history file (specific case)
  python example_usage.py --cice-real-case

  # Train with any NetCDF file
  python example_usage.py --netcdf-file /path/to/ocean_data.nc

  # Train with pre-processed data
  python example_usage.py --data-file training_data.npz
"""

import subprocess
import sys
from pathlib import Path


def run_with_sample_data() -> None:
    """Example: Create synthetic data and train."""
    print("=" * 60)
    print("Example 1: Training with synthetic data")
    print("=" * 60)

    cmd = [
        sys.executable, "../scripts/train.py",
        "--config", "../configs/config.yaml",
        "--create-data"
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_with_netcdf(netcdf_file: str) -> None:
    """Example: Train with NetCDF data."""
    print("=" * 60)
    print(f"Example 2: Training with NetCDF data from {netcdf_file}")
    print("=" * 60)

    cmd = [
        sys.executable, "../scripts/train.py",
        "--config", "../configs/config.yaml",
        "--netcdf-file", netcdf_file
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def convert_netcdf_only(netcdf_file: str, output_file: str) -> None:
    """Example: Convert NetCDF to training format without training."""
    print("=" * 60)
    print(f"Example 3: Converting {netcdf_file} to {output_file}")
    print("=" * 60)

    # Check if file exists
    if not Path(netcdf_file).exists():
        print(f"ERROR: NetCDF file not found: {netcdf_file}")
        return

    try:
        # Import required modules
        from icenet.data import IceDataPreparer
        from icenet.training import create_default_config

        # Create configuration and data preparer
        config = create_default_config()
        preparer = IceDataPreparer(config)

        print("Starting NetCDF conversion...")
        print(f"Input file: {netcdf_file}")
        print(f"Output file: {output_file}")

        # Perform conversion
        result = preparer.prepare_training_data(
            netcdf_file,
            output_file=output_file
        )

        print("\n✅ Conversion successful!")
        print("📊 Processed data summary:")
        print(f"   Training patterns: {result['inputs'].shape[0]:,}")
        print(f"   Features: {result['inputs'].shape[1]}")
        print(f"   Geographic coverage: {len(result['lons']):,} points")
        print(f"   Output saved to: {output_file}")

        # Show normalization statistics
        print("\n🔧 Normalization statistics:")
        feature_names = ['tair', 'tsfc', 'sst', 'sss', 'hs', 'hi', 'sice',
                         'lat']
        for i, name in enumerate(feature_names):
            mean_val = result['input_mean'][i]
            std_val = result['input_std'][i]
            print(f"   {name}: mean={mean_val:.3f}, std={std_val:.3f}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure the icenet package is properly installed")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


def analyze_netcdf_structure(netcdf_file: str) -> None:
    """Analyze NetCDF file structure and show variable compatibility."""
    print("=" * 60)
    print(f"Analyzing NetCDF file: {netcdf_file}")
    print("=" * 60)

    # Check if file exists
    if not Path(netcdf_file).exists():
        print(f"ERROR: NetCDF file not found: {netcdf_file}")
        return

    try:
        import netCDF4

        with netCDF4.Dataset(netcdf_file, 'r') as nc:
            # Find spatial dimensions (look for common patterns)
            spatial_dims = []
            time_dims = []

            for dim_name, dim in nc.dimensions.items():
                if any(pattern in dim_name.lower()
                       for pattern in ['ni', 'nx', 'xaxis']):
                    spatial_dims.append((dim_name, dim.size, 'longitude'))
                elif any(pattern in dim_name.lower()
                         for pattern in ['nj', 'ny', 'yaxis']):
                    spatial_dims.append((dim_name, dim.size, 'latitude'))
                elif (any(pattern in dim_name.lower()
                          for pattern in ['x', 'lon']) and
                      'axis' not in dim_name.lower()):
                    spatial_dims.append((dim_name, dim.size, 'longitude'))
                elif (any(pattern in dim_name.lower()
                          for pattern in ['y', 'lat']) and
                      'axis' not in dim_name.lower()):
                    spatial_dims.append((dim_name, dim.size, 'latitude'))
                elif any(pattern in dim_name.lower()
                         for pattern in ['time', 't']):
                    time_dims.append((dim_name, dim.size))

            print("📐 Dimensions found:")
            for dim_name, dim_size, dim_type in spatial_dims:
                print(f"   {dim_name}: {dim_size} ({dim_type})")
            for dim_name, dim_size in time_dims:
                print(f"   {dim_name}: {dim_size} (time)")

            # Try to get grid size info
            if len(spatial_dims) >= 2:
                x_dim = next((d for d in spatial_dims if 'lon' in d[2]),
                             spatial_dims[0])
                y_dim = next((d for d in spatial_dims if 'lat' in d[2]),
                             spatial_dims[1])
                print(f"🌍 Grid dimensions: {x_dim[1]} x {y_dim[1]}")

            if time_dims:
                print(f"⏰ Time steps: {time_dims[0][1]}")
            else:
                print("📊 No time dimension found (static file)")

            # Check required variables with flexible naming
            required_vars = [
                'Tair_h', 'Tsfc_h', 'sst_h', 'sss_h',
                'hs_h', 'hi_h', 'sice_h', 'aice_h', 'ULAT', 'umask'
            ]

            # Also check for alternative variable names
            alt_var_names = {
                'Tair_h': ['tair', 'air_temperature', 'T_air'],
                'Tsfc_h': ['tsfc', 'surface_temperature', 'T_sfc'],
                'sst_h': ['sst', 'sea_surface_temperature'],
                'sss_h': ['sss', 'sea_surface_salinity'],
                'hs_h': ['hs', 'snow_thickness', 'snow_depth', 'hsnon'],
                'hi_h': ['hi', 'ice_thickness', 'hicen'],
                'sice_h': ['sice', 'ice_salinity'],
                'aice_h': ['aice', 'ice_concentration', 'ice_fraction',
                           'aicen'],
                'ULAT': ['lat', 'latitude', 'TLAT'],
                'umask': ['mask', 'land_mask', 'ocean_mask']
            }

            print("\n🔍 Variable compatibility check:")
            missing_vars = []
            found_vars = []
            for var in required_vars:
                if var in nc.variables:
                    found_vars.append(var)
                    print(f"   ✓ Found {var}: {nc.variables[var].shape}")
                else:
                    # Check alternative names
                    alt_found = None
                    for alt_name in alt_var_names.get(var, []):
                        if alt_name in nc.variables:
                            alt_found = alt_name
                            found_vars.append(f"{var} (as {alt_name})")
                            shape_info = nc.variables[alt_name].shape
                            print(f"   ✓ Found {var} as {alt_name}: "
                                  f"{shape_info}")
                            break

                    if not alt_found:
                        missing_vars.append(var)

            if missing_vars:
                print(f"\n⚠️  Missing variables: {missing_vars}")
                print("\n📋 Available variables in file:")
                # Show first 20 variables
                for var_name in sorted(nc.variables.keys())[:20]:
                    var = nc.variables[var_name]
                    print(f"   {var_name}: {var.shape}")
                if len(nc.variables) > 20:
                    print(f"   ... and {len(nc.variables) - 20} more "
                          f"variables")
            else:
                print("\n✅ All required variables found!")
                print("🚀 File is ready for IceNet training!")

    except ImportError:
        print("❌ WARNING: netCDF4 not available, skipping file analysis")
    except Exception as e:
        print(f"❌ ERROR reading NetCDF file: {e}")


def run_cice_real_case() -> None:
    """Example: Train with real CICE history NetCDF file."""
    # cice_file = ("/home/gvernier/sandboxes/jedi-bundle-new/build/soca/"
    #              "test/data_static/72x35x25/history/cice_history.nc")
    cice_file = ("/home/gvernier/data/gdas.agg_ice.t00z.inst.f009.nc")
    print("=" * 60)
    print("Example 4: Training with real CICE history data")
    print(f"File: {cice_file}")
    print("=" * 60)

    # Check if file exists
    if not Path(cice_file).exists():
        print(f"ERROR: NetCDF file not found: {cice_file}")
        print("Please ensure the GDAS file is available")
        return

    # First, analyze the NetCDF file
    print("Analyzing NetCDF file structure...")
    analyze_netcdf_structure(cice_file)

    print("\n" + "=" * 40)
    print("Starting training process...")
    print("=" * 40)

    # Run training with the CICE file
    cmd = [
        sys.executable, "../scripts/train.py",
        "--config", "../configs/config.yaml",
        "--netcdf-file", cice_file
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print("This will:")
    print("1. Read the CICE NetCDF file")
    print("2. Extract training variables (7 features + ice concentration)")
    print("3. Apply domain filtering (Arctic/Antarctic)")
    print("4. Create train/validation split")
    print("5. Train the IceNet model")
    print("6. Save model and training plots")
    print("\nStarting training...")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("SUCCESS: Training completed with real CICE data!")
        print("Check the following outputs:")
        print("  - Model: models/best_model.pt")
        print("  - Data: data/cice_training_data.npz")
        print("  - Plot: models/training_history.png")
        print("=" * 60)
    else:
        print(f"\nERROR: Training failed with exit code "
              f"{result.returncode}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='IceNet training examples')
    parser.add_argument('--create-sample-data', action='store_true',
                        help='Create and train with synthetic data')
    parser.add_argument('--netcdf-file', type=str,
                        help='NetCDF file to use for training')
    parser.add_argument('--cice-real-case', action='store_true',
                        help='Train with real CICE history NetCDF file')
    parser.add_argument('--data-file', type=str,
                        help='Pre-processed data file to use for training')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only convert NetCDF, do not train')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze NetCDF structure, do not convert')
    parser.add_argument('--output-file', type=str,
                        default='converted_data.npz',
                        help='Output file for conversion')

    args = parser.parse_args()

    if args.create_sample_data:
        run_with_sample_data()
    elif args.cice_real_case:
        run_cice_real_case()
    elif args.netcdf_file:
        if args.convert_only:
            convert_netcdf_only(args.netcdf_file, args.output_file)
        elif args.analyze_only:
            analyze_netcdf_structure(args.netcdf_file)
        else:
            run_with_netcdf(args.netcdf_file)
    elif args.data_file:
        print("=" * 60)
        print(f"Training with pre-processed data: {args.data_file}")
        print("=" * 60)

        cmd = [
            sys.executable, "../scripts/train.py",
            "--config", "../configs/config.yaml",
            "--data-path", args.data_file
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print("Please specify one of:")
        print("  --create-sample-data")
        print("  --cice-real-case")
        print("  --netcdf-file <file.nc>")
        print("  --netcdf-file <file.nc> --convert-only")
        print("  --netcdf-file <file.nc> --analyze-only")
        print("  --data-file <file.npz>")
        print("\nFor help: python example_usage.py --help")


if __name__ == "__main__":
    main()
