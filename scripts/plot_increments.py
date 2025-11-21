#!/usr/bin/env python3
"""
IceNet Data Assimilation Increment Visualization Script

This script:
1. Reads NetCDF data and computes Jacobian matrices using trained IceNet model
2. Reconstructs total increments using B = KK^T formulation
3. Assumes unit increment in ice concentration observation
4. Plots increments for ice thickness (hi) and snow depth (hs)

The increment reconstruction follows the data assimilation formula:
    x^a = x^b + Δx

Where the increment is computed using:
    Δx = BH^T(HBH^T + R)^(-1)(y^o - H(x^b))

Assuming B = KD²K^T where D² is a variance matrix, and for a unit ice concentration increment:
    Δx = KD²K^T * innovation = K * D² * innovation

Usage:
    python plot_increments.py --netcdf-file data.nc --model models/best_model.pt
    python plot_increments.py --netcdf-file data.nc --model models/best_model.pt --arctic-only
    python plot_increments.py --netcdf-file data.nc --model models/best_model.pt --antarctic-only
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import icenet modules
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import torch
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

# Cartopy for polar projections
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from icenet.model import IceNet
from icenet.data import select_data


class IncrementPlotter:
    """Handles increment reconstruction and visualization for data assimilation."""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """Initialize with model and configuration."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Initialize feature names (will be set during model loading)
        self.feature_names = []
        self.feature_units = []

        # Variable mappings (same as in plot_inference_jacobian.py)
        self.var_names = {
            "lat": "ULAT",
            "lon": "ULON",
            "aice": "aice_h",
            "tsfc": "Tsfc_h",
            "sst": "sst_h",
            "sss": "sss_h",
            "sice": "sice_h",
            "hi": "hi_h",
            "hs": "hs_h",
            "mask": "umask",
            "tair": "Tair_h",
            "frzmlt": "frzmlt_h",
            "strocnx": "strocnx_h",
            "strocny": "strocny_h",
            "strairx": "strairx_h",
            "strairy": "strairy_h",
            #"fhocn": "fhocn_h",
            "qref": "Qref_h",
            #"flwup": "flwup_h",
            #"fsens": "fsens_h",
            "flat": "flat_h",
            "flwdn": "flwdn_h",
            "fswdn": "fswdn_h",
        }

        # Alternative variable names for GDAS compatibility
        self.alt_var_names = {
            "aice_h": ["aice", "ice_concentration", "aicen"],
            "hi_h": ["hi", "ice_thickness", "hicen"],
            "hs_h": ["hs", "snow_thickness", "hsnon"],
            "Tair_h": ["tair", "air_temperature"],
            "Tsfc_h": ["tsfc", "surface_temperature"],
            "sst_h": ["sst", "sea_surface_temperature"],
            "sss_h": ["sss", "sea_surface_salinity"],
            "sice_h": ["sice", "ice_salinity"],
            "frzmlt_h": ["frzmlt", "frazil_melt", "frazil_ice_melt"],
            "ULAT": ["lat", "latitude"],
            "ULON": ["lon", "longitude"],
            "umask": ["mask", "land_mask"],
        }

        # Load model and extract configuration
        self.model, self.config = self._load_model_and_config(model_path, config_path)

    def _load_model_and_config(
        self, model_path: str, config_path: Optional[str] = None
    ) -> Tuple[IceNet, Dict]:
        """Load trained IceNet model and configuration."""
        print(f"Loading model from: {model_path}")

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get configuration from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
            model_config = config["model"]
        else:
            raise ValueError(
                "No config found in checkpoint - cannot determine model architecture"
            )

        # Create model with saved configuration
        model = IceNet(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            output_size=model_config["output_size"],
            hidden_layers=model_config.get("hidden_layers", 2)
        )

        # Load model weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        # Load normalization parameters
        try:
            norm_paths = [
                Path(model_path).parent / "normalization.pt",
                Path(model_path).parent / "normalization.normalization.pt",
                Path(model_path).parent / f"normalization.{Path(model_path).stem}.pt"
            ]

            loaded = False
            for norm_path in norm_paths:
                if norm_path.exists():
                    moments = torch.load(norm_path, map_location=self.device, weights_only=False)
                    model.input_mean.data = moments[0]
                    model.input_std.data = moments[1]
                    print(f"✅ Loaded normalization from: {norm_path}")
                    loaded = True
                    break

            if not loaded:
                print("⚠️ No normalization file found, using default normalization")
                input_size = model_config["input_size"]
                model.input_mean.data = torch.zeros(input_size)
                model.input_std.data = torch.ones(input_size)

        except Exception as e:
            print(f"Warning: Could not load normalization: {e}")
            input_size = model_config["input_size"]
            model.input_mean.data = torch.zeros(input_size)
            model.input_std.data = torch.ones(input_size)

        print("✅ Model loaded successfully")

        # Set up feature names based on input_size
        self._setup_features(model_config["input_size"])

        return model, config

    def _setup_features(self, input_size: int):
        """Set up feature names and units based on model input size."""
        print(f"Setting up features for input_size: {input_size}")

        # Standard feature order matching training data preparation
        all_features = [
            # Basic ocean and atmosphere variables
            ("SST", "°C"),
            ("SSS", "psu"),
            ("Tair", "°C"),
            ("Tsfc", "°C"),
            ("hi", "m"),      # Ice thickness - index 4
            ("hs", "m"),      # Snow depth - index 5
            ("sice", "psu"),
            # Stress variables
            ("strocnx", "N/m²"),
            ("strocny", "N/m²"),
            ("strairx", "N/m²"),
            ("strairy", "N/m²"),
            # Heat flux variables
            ("qref", "kg/kg"),
            ("flwdn", "W/m²"),
            ("fswdn", "W/m²")
        ]

        if input_size > len(all_features):
            raise ValueError(f"Input size {input_size} exceeds available features {len(all_features)}")

        if input_size <= 0:
            print(f"WARNING: input_size is {input_size}, defaulting to 1 (SST only)")
            input_size = 1

        self.feature_names = [name for name, _ in all_features[:input_size]]
        self.feature_units = [unit for _, unit in all_features[:input_size]]

        print(f"Model expects {input_size} features: {self.feature_names}")

        # Find indices for ice thickness and snow depth
        self.hi_index = None
        self.hs_index = None

        for i, name in enumerate(self.feature_names):
            if name == "hi":
                self.hi_index = i
                print(f"✅ Ice thickness (hi) found at index {i}")
            elif name == "hs":
                self.hs_index = i
                print(f"✅ Snow depth (hs) found at index {i}")

        if self.hi_index is None:
            print("⚠️ Ice thickness (hi) not available in model features")
        if self.hs_index is None:
            print("⚠️ Snow depth (hs) not available in model features")

    def read_netcdf_data(self, filename: str) -> Dict[str, np.ndarray]:
        """Read NetCDF data with variable name fallback."""
        print(f"Reading NetCDF file: {filename}")

        with nc.Dataset(filename, "r") as dataset:
            data = {}

            for key, var_name in self.var_names.items():
                found_var = None

                # Try primary variable name
                if var_name in dataset.variables:
                    found_var = var_name
                else:
                    # Try alternative names
                    alt_names = self.alt_var_names.get(var_name, [])
                    for alt_name in alt_names:
                        if alt_name in dataset.variables:
                            found_var = alt_name
                            print(f"Using alternative: {var_name} -> {alt_name}")
                            break

                if found_var:
                    var_data = dataset.variables[found_var][:]
                    # Handle time dimension if present
                    if var_data.ndim == 3:  # (time, lat, lon)
                        var_data = var_data[0]  # Take first time step
                    data[key] = var_data
                    print(f"Read {key}: shape {data[key].shape}")
                else:
                    # Handle missing variables gracefully for optional features
                    optional_vars = ["frzmlt", "sice", "hs", "strocnx", "strocny", "strairx", "strairy",
                                   "qref", "flwup", "flwdn", "fswdn"]
                    if key in optional_vars:
                        print(f"⚠️ Optional variable {var_name} not found - will use zeros if needed")
                    else:
                        available_vars = list(dataset.variables.keys())[:10]
                        raise KeyError(
                            f"Required variable {var_name} not found. "
                            f"Available: {available_vars}"
                        )

        return data

    def filter_domain(
        self, data: Dict[str, np.ndarray], domain: str = "arctic",
        clean_data: bool = True
    ) -> Tuple[np.ndarray, ...]:
        """Filter data for Arctic or Antarctic domain using training criteria."""
        lats = data["lat"]
        lons = data["lon"]
        mask = data["mask"]

        # Flatten arrays
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        mask_flat = mask.flatten()

        # Get other variables flattened
        aice_flat = data["aice"].flatten()
        sst_flat = data["sst"].flatten()

        # Use the same filtering logic as training data
        if domain.lower() == "arctic":
            pole = "north"
        elif domain.lower() == "antarctic":
            pole = "south"
        else:
            raise ValueError("Domain must be 'arctic' or 'antarctic'")

        # Apply the same select_data filter used in training
        selected_indices = []
        for i in range(len(lats_flat)):
            if select_data(
                float(mask_flat[i]),
                float(lats_flat[i]),
                float(aice_flat[i]),
                float(sst_flat[i]),
                clean_data,
                pole
            ):
                selected_indices.append(i)

        print(f"Selected {len(selected_indices)} points for {domain} domain "
              f"using training filter criteria (clean_data={clean_data})")

        if len(selected_indices) == 0:
            raise ValueError(f"No valid data points found for {domain} domain")

        # Create domain mask
        domain_mask = np.zeros(len(lats_flat), dtype=bool)
        domain_mask[selected_indices] = True

        # Extract features for valid points based on model requirements
        features = []
        feature_var_mapping = {
            "SST": "sst",
            "SSS": "sss",
            "Tair": "tair",
            "Tsfc": "tsfc",
            "hi": "hi",
            "hs": "hs",
            "sice": "sice",
            "strocnx": "strocnx",
            "strocny": "strocny",
            "strairx": "strairx",
            "strairy": "strairy",
            "qref": "qref",
            "flwdn": "flwdn",
            "fswdn": "fswdn"
        }

        for feature_name in self.feature_names:
            if feature_name in feature_var_mapping:
                var_key = feature_var_mapping[feature_name]
                if var_key in data:
                    var_data = data[var_key].flatten()[domain_mask]
                    features.append(var_data)
                    print(f"✅ Added feature {feature_name} ({var_key}), shape: {var_data.shape}")
                else:
                    # Handle missing features by filling with zeros
                    print(f"⚠️ Feature {feature_name} ({var_key}) not found in data - filling with zeros")
                    n_points = np.sum(domain_mask)
                    zero_data = np.zeros(n_points)
                    features.append(zero_data)
                    print(f"✅ Added zero-filled feature {feature_name}, shape: {zero_data.shape}")
            else:
                raise ValueError(f"Unknown feature name: {feature_name}")

        if len(features) == 0:
            raise ValueError("No features were successfully extracted")

        features = np.column_stack(features)
        print(f"Final features shape: {features.shape}")
        targets = aice_flat[domain_mask]

        return (
            features,
            targets,
            lons_flat[domain_mask],
            lats_flat[domain_mask],
            domain_mask,
            lats.shape,
        )

    def compute_jacobians(self, features: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix (dH/dx) for all samples."""
        print("Computing Jacobians...")

        # Convert to torch tensors
        features_tensor = torch.FloatTensor(features).to(self.device)
        jacobians = []

        # Compute Jacobian for each sample
        for i in range(len(features)):
            sample = features_tensor[i:i + 1]  # Single sample
            sample.requires_grad_(True)

            # Forward pass
            output = self.model(sample)

            # Compute gradients (dH/dx)
            output.backward()
            jac = sample.grad.cpu().numpy().flatten()
            jacobians.append(jac)

            # Clear gradients
            sample.grad = None

        jacobians = np.array(jacobians)
        print(f"✅ Jacobian computation complete: shape {jacobians.shape}")
        return jacobians

    def compute_increments(
        self, jacobians: np.ndarray, unit_increment: float = 1.0
    ) -> np.ndarray:
        """
        Compute data assimilation increments assuming B = KD²K^T where D² is a variance matrix.

        The increment formula is:
        Δx = BH^T(HBH^T + R)^(-1)(y^o - H(x^b))

        For B = KD²K^T and assuming R is small compared to HBH^T:
        Δx ≈ KD²K^T * H^T * (HKD² K^TH^T)^(-1) * innovation

        Since H is the identity for ice concentration observations:
        Δx = KD² * innovation

        Args:
            jacobians: Jacobian matrix K (n_points, n_features)
            unit_increment: Observation increment in ice concentration (default: 1.0)

        Returns:
            increments: State increments (n_points, n_features)
        """
        print(f"Computing increments with unit increment = {unit_increment}")
        print(f"Using B = KD²K^T formulation where D² is a variance matrix")

        n_points, n_features = jacobians.shape

        # Create variance matrix D² = I (identity matrix)
        D_squared = np.eye(n_features)
        print(f"Using variance matrix D² = I (identity matrix)")
        print(f"D² diagonal values: 1.0 (constant)")

        # Compute increments: Δx = K * D² * innovation
        # For each point, we compute K_i * D² * innovation
        increments = np.zeros_like(jacobians)

        for i in range(n_points):
            # K_i is a row vector (1, n_features)
            # D² is (n_features, n_features)
            # K_i * D² gives (1, n_features)
            k_i = jacobians[i:i+1, :]  # Keep as (1, n_features)
            increment_i = np.dot(k_i, D_squared) * unit_increment  # (1, n_features)
            increments[i, :] = increment_i.flatten()

        print(f"✅ Increment computation complete: shape {increments.shape}")
        print(f"D² matrix stats - Mean: {np.mean(D_squared):.6f}, Std: {np.std(D_squared):.6f}")
        return increments

    def create_colormap(self, name: str, centered: bool = True):
        """Create custom colormaps for different variables."""
        if centered:
            # Centered colormap for increments (blue-white-red)
            return plt.cm.RdBu_r
        else:
            # Default colormap
            return plt.cm.viridis

    def plot_increment_fields(
        self,
        increments: np.ndarray,
        features: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        domain: str,
        unit_increment: float,
        output_dir: str = "plots",
    ):
        """Plot ice thickness and snow depth increments."""

        Path(output_dir).mkdir(exist_ok=True)

        # Compute ice concentration increment by applying the increments through the model
        # Since increment = jacobian * unit_increment, the resulting aice increment is:
        # Δaice = Σ(∂aice/∂xi * Δxi) = Σ(jacobian_i * increment_i) = unit_increment (by construction)
        # But we can also compute it from the individual increments and jacobians

        # For meaningful correlations, compute the correlation between:
        # 1. Ice concentration increment (Δaice) and ice thickness increment (Δhi)
        # 2. Ice concentration increment (Δaice) and snow depth increment (Δhs)

        aice_hi_corr = None
        aice_hs_corr = None

        # The ice concentration increment is unit_increment for all points (by construction)
        # so correlation with other increments is just the correlation of those increments with a constant
        # This would be undefined, so let's compute the actual aice increment from the model

        # Compute aice increment by forward propagating the input increments through the model
        features_tensor = torch.FloatTensor(features).to(self.device)
        increments_tensor = torch.FloatTensor(increments).to(self.device)

        # Compute baseline aice prediction
        with torch.no_grad():
            baseline_aice = self.model(features_tensor).cpu().numpy().flatten()

        # Compute aice prediction with increments added
        features_with_increments = features_tensor + increments_tensor
        with torch.no_grad():
            perturbed_aice = self.model(features_with_increments).cpu().numpy().flatten()

        # Ice concentration increment
        aice_increment = perturbed_aice - baseline_aice

        # Now compute meaningful correlations
        if self.hi_index is not None:
            hi_increment = increments[:, self.hi_index]
            if np.std(hi_increment) > 1e-10 and np.std(aice_increment) > 1e-10:
                aice_hi_corr = np.corrcoef(aice_increment, hi_increment)[0, 1]

        if self.hs_index is not None:
            hs_increment = increments[:, self.hs_index]
            if np.std(hs_increment) > 1e-10 and np.std(aice_increment) > 1e-10:
                aice_hs_corr = np.corrcoef(aice_increment, hs_increment)[0, 1]

        # Choose projection based on domain
        if domain.lower() == "arctic":
            projection = ccrs.NorthPolarStereo()
            extent = [-180, 180, 50, 90]
        else:  # antarctic
            projection = ccrs.SouthPolarStereo()
            extent = [-180, 180, -90, -50]
        transform = ccrs.PlateCarree()

        # Create figure for increment plots
        fig = plt.figure(figsize=(16, 8))

        plots_created = 0

        # Plot ice thickness increment if available
        if self.hi_index is not None:
            hi_increment = increments[:, self.hi_index]

            ax1 = plt.subplot(1, 2, 1, projection=projection)
            ax1.set_extent(extent, crs=ccrs.PlateCarree())
            ax1.add_feature(cfeature.COASTLINE, alpha=0.5)
            ax1.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
            ax1.gridlines(draw_labels=True, alpha=0.3)

            # Calculate symmetric color range limited to 1/3 of max absolute value
            std_dev = np.std(hi_increment)
            vmax = std_dev
            vmin = -vmax

            scatter1 = ax1.scatter(
                lons, lats, c=hi_increment, s=1.0,
                cmap=self.create_colormap("increment", centered=True),
                vmin=vmin, vmax=vmax,
                transform=transform, alpha=0.7
            )

            # Add title as text box
            hi_title = (f"Ice Thickness Increment\n"
                       f"Data range: [{np.min(hi_increment):.4f}, {np.max(hi_increment):.4f}] m\n"
                       f"Color range: [{vmin:.4f}, {vmax:.4f}] m (1/3 max)\n"
                       f"Unit ice conc. increment: {unit_increment}")
            if aice_hi_corr is not None:
                hi_title += f"\nCorr(Δaice, Δhi): {aice_hi_corr:.3f}"
            ax1.text(0.02, 0.98, hi_title, transform=ax1.transAxes,
                    fontsize=9, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            plt.colorbar(scatter1, ax=ax1, shrink=0.6, label="Δhi (m)")
            plots_created += 1

        # Plot snow depth increment if available
        if self.hs_index is not None:
            hs_increment = increments[:, self.hs_index]

            subplot_pos = 2 if plots_created == 1 else 1
            ax2 = plt.subplot(1, 2, subplot_pos, projection=projection)
            ax2.set_extent(extent, crs=ccrs.PlateCarree())
            ax2.add_feature(cfeature.COASTLINE, alpha=0.5)
            ax2.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
            ax2.gridlines(draw_labels=True, alpha=0.3)

            # Calculate symmetric color range limited to 1/10 of max absolute value
            std_dev = np.std(hs_increment)
            vmax = std_dev
            vmin = -vmax

            scatter2 = ax2.scatter(
                lons, lats, c=hs_increment, s=1.0,
                cmap=self.create_colormap("increment", centered=True),
                vmin=vmin, vmax=vmax,
                transform=transform, alpha=0.7
            )

            # Add title as text box
            hs_title = (f"Snow Depth Increment\n"
                       f"Data range: [{np.min(hs_increment):.4f}, {np.max(hs_increment):.4f}] m\n"
                       f"Color range: [{vmin:.4f}, {vmax:.4f}] m (1/3 max)\n"
                       f"Unit ice conc. increment: {unit_increment}")
            if aice_hs_corr is not None:
                hs_title += f"\nCorr(Δaice, Δhs): {aice_hs_corr:.3f}"
            ax2.text(0.02, 0.98, hs_title, transform=ax2.transAxes,
                    fontsize=9, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            plt.colorbar(scatter2, ax=ax2, shrink=0.6, label="Δhs (m)")
            plots_created += 1

        if plots_created == 0:
            print("⚠️ Neither ice thickness nor snow depth available for plotting")
            return None

        # Add main title
        main_title = (f"Data Assimilation Increments Proxy - {domain.title()} Domain\n"
                     f"B = KK^T, Unit ice concentration increment = {unit_increment}")
        plt.figtext(0.5, 0.95, main_title,
                   fontsize=14, fontweight="bold",
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save plot
        output_file = f"{output_dir}/da_increments_{domain.lower()}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✅ Saved increment plot: {output_file}")

        return fig

    def create_increment_statistics(
        self,
        increments: np.ndarray,
        features: np.ndarray,
        domain: str,
        unit_increment: float,
    ):
        """Print increment statistics."""
        print(f"\n📊 {domain.title()} Domain Increment Statistics:")
        print("=" * 50)
        print(f"Unit ice concentration increment: {unit_increment}")
        print(f"Number of grid points: {increments.shape[0]}")
        print(f"Number of features: {increments.shape[1]}")

        # Ice thickness increment statistics
        if self.hi_index is not None:
            hi_increment = increments[:, self.hi_index]
            hi_background = features[:, self.hi_index]

            print(f"\n🧊 Ice Thickness (hi) Increments:")
            print(f"  Range: [{np.min(hi_increment):.6f}, {np.max(hi_increment):.6f}] m")
            print(f"  Mean: {np.mean(hi_increment):.6f} m")
            print(f"  Std:  {np.std(hi_increment):.6f} m")
            print(f"  RMS:  {np.sqrt(np.mean(hi_increment**2)):.6f} m")

            # Relative increment statistics
            non_zero_bg = hi_background[hi_background > 0.01]  # Avoid division by zero
            if len(non_zero_bg) > 0:
                hi_inc_nz = hi_increment[hi_background > 0.01]
                rel_increment = hi_inc_nz / non_zero_bg * 100
                print(f"  Relative increment (where hi > 0.01m):")
                print(f"    Range: [{np.min(rel_increment):.2f}, {np.max(rel_increment):.2f}] %")
                print(f"    Mean: {np.mean(rel_increment):.2f} %")

            # Background field statistics for context
            print(f"  Background hi statistics:")
            print(f"    Range: [{np.min(hi_background):.3f}, {np.max(hi_background):.3f}] m")
            print(f"    Mean: {np.mean(hi_background):.3f} m")

        # Snow depth increment statistics
        if self.hs_index is not None:
            hs_increment = increments[:, self.hs_index]
            hs_background = features[:, self.hs_index]

            print(f"\n❄️ Snow Depth (hs) Increments:")
            print(f"  Range: [{np.min(hs_increment):.6f}, {np.max(hs_increment):.6f}] m")
            print(f"  Mean: {np.mean(hs_increment):.6f} m")
            print(f"  Std:  {np.std(hs_increment):.6f} m")
            print(f"  RMS:  {np.sqrt(np.mean(hs_increment**2)):.6f} m")

            # Relative increment statistics
            non_zero_bg = hs_background[hs_background > 0.001]  # Avoid division by zero
            if len(non_zero_bg) > 0:
                hs_inc_nz = hs_increment[hs_background > 0.001]
                rel_increment = hs_inc_nz / non_zero_bg * 100
                print(f"  Relative increment (where hs > 0.001m):")
                print(f"    Range: [{np.min(rel_increment):.2f}, {np.max(rel_increment):.2f}] %")
                print(f"    Mean: {np.mean(rel_increment):.2f} %")

            # Background field statistics for context
            print(f"  Background hs statistics:")
            print(f"    Range: [{np.min(hs_background):.3f}, {np.max(hs_background):.3f}] m")
            print(f"    Mean: {np.mean(hs_background):.3f} m")




def thin_data(features, targets, lons, lats, increments, fraction):
    """Thin data for plotting if needed."""
    if fraction < 1.0:
        n = len(targets)
        n_thin = int(n * fraction)
        idx = np.random.choice(n, n_thin, replace=False)
        return features[idx], targets[idx], lons[idx], lats[idx], increments[idx]
    return features, targets, lons, lats, increments


def main():
    parser = argparse.ArgumentParser(
        description="IceNet Data Assimilation Increment Visualization"
    )
    parser.add_argument(
        "--netcdf-file", required=True, help="Input NetCDF file"
    )
    parser.add_argument(
        "--model", required=True, help="Trained model file (.pt)"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Config file"
    )
    parser.add_argument(
        "--output-dir", default="plots", help="Output directory"
    )
    parser.add_argument(
        "--arctic-only", action="store_true", help="Plot Arctic only"
    )
    parser.add_argument(
        "--antarctic-only", action="store_true", help="Plot Antarctic only"
    )
    parser.add_argument(
        '--thin-fraction', type=float, default=1.0,
        help='Fraction of data to plot (e.g. 0.5 for 50 percent)'
    )
    parser.add_argument(
        '--unit-increment', type=float, default=1.0,
        help='Unit increment in ice concentration for DA (default: 1.0)'
    )


    args = parser.parse_args()

    # Initialize plotter
    plotter = IncrementPlotter(args.model, args.config)

    # Check if ice thickness and snow depth are available
    if plotter.hi_index is None and plotter.hs_index is None:
        print("❌ Neither ice thickness (hi) nor snow depth (hs) available in model features")
        print(f"Available features: {plotter.feature_names}")
        return

    # Read NetCDF data
    data = plotter.read_netcdf_data(args.netcdf_file)

    # Determine domains to process
    domains = []
    if args.arctic_only:
        domains = ["arctic"]
    elif args.antarctic_only:
        domains = ["antarctic"]
    else:
        domains = ["arctic", "antarctic"]

    # Process each domain
    for domain in domains:
        print(f"\n🌍 Processing {domain.title()} domain...")

        try:
            # Filter data for domain
            features, targets, lons, lats, mask, shape = plotter.filter_domain(
                data, domain, clean_data=True
            )

            if len(features) == 0:
                print(f"⚠️ No valid data points found for {domain} domain")
                continue

            # Compute Jacobian matrix K
            jacobians = plotter.compute_jacobians(features)

            # Compute increments using B = KD²K^T formulation
            increments = plotter.compute_increments(jacobians, args.unit_increment)

            # Thin data for plotting if requested
            features, targets, lons, lats, increments = thin_data(
                features, targets, lons, lats, increments, args.thin_fraction
            )

            # Create increment plots
            fig = plotter.plot_increment_fields(
                increments,
                features,
                lons,
                lats,
                domain,
                args.unit_increment,
                args.output_dir,
            )

            # Print statistics
            plotter.create_increment_statistics(
                increments, features, domain, args.unit_increment
            )

            if fig:
                plt.close(fig)  # Free memory

        except Exception as e:
            print(f"❌ Error processing {domain} domain: {e}")
            continue

    print(f"\n✅ Increment analysis complete! Check {args.output_dir}/ for plots")


if __name__ == "__main__":
    main()
