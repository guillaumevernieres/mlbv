#!/usr/bin/env python3
"""
Example script demonstrating how to use the IceNet Python training system
with NetCDF data preparation.

Usage examples:
  # Create and train with synthetic data
  python example_usage.py --create-sample-data

  # Train with NetCDF file
  python example_usage.py --netcdf-file /path/to/ocean_data.nc

  # Train with pre-processed data
  python example_usage.py --data-file training_data.npz
"""

import subprocess
import sys


def run_with_sample_data():
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


def run_with_netcdf(netcdf_file):
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


def convert_netcdf_only(netcdf_file, output_file):
    """Example: Convert NetCDF to training format without training."""
    print("=" * 60)
    print(f"Example 3: Converting {netcdf_file} to {output_file}")
    print("=" * 60)

    # Note: For demonstration - would use icenet.data module directly
    print("Would use: from icenet.data import create_training_data_from_netcdf")
    print(f"create_training_data_from_netcdf('{netcdf_file}', config, '{output_file}')")
    print("Conversion functionality is available through the icenet.data module")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='IceNet training examples')
    parser.add_argument('--create-sample-data', action='store_true',
                        help='Create and train with synthetic data')
    parser.add_argument('--netcdf-file', type=str,
                        help='NetCDF file to use for training')
    parser.add_argument('--data-file', type=str,
                        help='Pre-processed data file to use for training')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only convert NetCDF, do not train')
    parser.add_argument('--output-file', type=str,
                        default='converted_data.npz',
                        help='Output file for conversion')

    args = parser.parse_args()

    if args.create_sample_data:
        run_with_sample_data()
    elif args.netcdf_file:
        if args.convert_only:
            convert_netcdf_only(args.netcdf_file, args.output_file)
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
        print("  --netcdf-file <file.nc>")
        print("  --data-file <file.npz>")
        print("\nFor help: python example_usage.py --help")


if __name__ == "__main__":
    main()
