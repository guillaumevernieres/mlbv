#!/usr/bin/env python3
"""
IceNet Inference and Jacobian Visualization Script

This script:
1. Reads NetCDF data with GDAS compatibility
2. Runs inference with trained IceNet model
3. Computes Jacobian matrices
4. Creates separate Arctic/Antarctic field plots
5. Shows input features, predictions, and Jacobian sensitivities

Usage:
    python plot_inference_jacobian.py --netcdf-file data.nc \\
        --model models/best_model.pt
    python plot_inference_jacobian.py --netcdf-file data.nc \\
        --model models/best_model.pt --arctic-only
    python plot_inference_jacobian.py --netcdf-file data.nc \\
        --model models/best_model.pt --antarctic-only
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Cartopy for polar projections
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from icenet.model import IceNet
from icenet.data import select_data


class IceNetInferencePlotter:
    """Handles NetCDF data processing, inference, and visualization."""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """Initialize with model and configuration."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Initialize feature names (will be set during model loading)
        self.feature_names = []
        self.feature_units = []

        # Variable mappings (same as in data.py)
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
            # Ice temperatures
            "sitempbot": "sitempbot_h",
            "sitempsnic": "sitempsnic_h",
            "sitemptop": "sitemptop_h",
            # Ocean/ice stresses
            "strocnx": "strocnx_h",
            "strocny": "strocny_h",
            # Atmosphere/ice stresses
            "strairx": "strairx_h",
            "strairy": "strairy_h",
            # Heat and flux variables
            "fhocn": "fhocn_h",
            "qref": "Qref_h",
            "flwup": "flwup_h",
            "fsens": "fsens_h",
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

        # Load model and extract configuration from checkpoint
        self.model, self.config = self._load_model_and_config(
            model_path, config_path
        )

        # Debug: Check if feature_names were properly set
        print(f"DEBUG after model loading: feature_names = {self.feature_names}")
        print(f"DEBUG after model loading: feature_units = {self.feature_units}")

    def _load_model_and_config(
        self, model_path: str, config_path: Optional[str] = None
    ) -> Tuple[IceNet, Dict]:
        """Load trained IceNet model and configuration."""
        print(f"Loading model from: {model_path}")

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Debug checkpoint contents
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        # Get configuration from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
            print(f"Full config: {config}")
            model_config = config["model"]
        else:
            print("No config found in checkpoint, checking for direct model_config...")
            # Try to find model config in other locations
            for key in checkpoint.keys():
                print(f"  {key}: {type(checkpoint[key])}")
            raise ValueError(
                "No config found in checkpoint - cannot determine model architecture"
            )

        # Debug model configuration
        print(f"Model config found: {model_config}")
        print(f"Input size: {model_config['input_size']}")

        # Debug: Print the expected feature list before model creation
        print(f"DEBUG: About to setup features for input_size {model_config['input_size']}")

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
            # Try different normalization file locations
            norm_paths = [
                Path(model_path).parent / "normalization.pt",
                Path(model_path).parent / "normalization.normalization.pt",
                Path(model_path).parent / f"normalization.{Path(model_path).stem}.pt"
            ]

            loaded = False
            for norm_path in norm_paths:
                if norm_path.exists():
                    moments = torch.load(norm_path, map_location=self.device)
                    model.input_mean.data = moments[0]
                    model.input_std.data = moments[1]
                    print(f"✅ Loaded normalization from: {norm_path}")
                    loaded = True
                    break

            if not loaded:
                print("⚠️ No normalization file found, trying model.load_norm...")
                model.load_norm(model_path)
                loaded = True

        except Exception as e:
            print(f"Warning: Could not load normalization: {e}")
            print("Using default normalization (mean=0, std=1)")
            # Set default normalization
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
            ("hi", "m"),
            ("hs", "m"),
            ("sice", "psu"),
            # Stress variables (ice temp vars removed due to NaN issues)
            ("strocnx", "N/m²"),
            ("strocny", "N/m²"),
            ("strairx", "N/m²"),
            ("strairy", "N/m²"),
            # Heat flux variables
            ("fhocn", "W/m²"),
            ("qref", "kg/kg"),
            ("fsens", "W/m²"),
            ("flat", "W/m²"),
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
        print(f"Features stored in self.feature_names: {self.feature_names}")
        print(f"Units stored in self.feature_units: {self.feature_units}")

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
                            print(
                                f"Using alternative: "
                                f"{var_name} -> {alt_name}"
                            )
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
                    optional_vars = ["frzmlt", "sice", "hs", "sitempbot", "sitempsnic",
                                   "sitemptop", "strocnx", "strocny", "strairx", "strairy",
                                   "fhocn", "qref", "flwup", "fsens", "flat", "flwdn", "fswdn"]
                    if key in optional_vars:  # Optional features
                        print(f"⚠️ Optional variable {var_name} not found - will use zeros if needed")
                        # Don't add to data dict - will be handled in feature extraction
                    else:
                        # Required variables (coordinates, basic fields)
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
        print(f"DEBUG: About to extract features...")
        print(f"DEBUG: self.feature_names = {self.feature_names}")
        print(f"DEBUG: len(self.feature_names) = {len(self.feature_names)}")
        print(f"Model expects features: {self.feature_names}")
        print(f"Available data keys: {list(data.keys())}")

        features = []
        feature_var_mapping = {
            "SST": "sst",
            "SSS": "sss",
            "Tair": "tair",
            "Tsfc": "tsfc",
            "hi": "hi",
            "hs": "hs",
            "sice": "sice",
            # Stress variables (ice temp vars removed due to NaN issues)
            "strocnx": "strocnx",
            "strocny": "strocny",
            "strairx": "strairx",
            "strairy": "strairy",
            # Heat flux variables
            "fhocn": "fhocn",
            "qref": "qref",
            "fsens": "fsens",
            "flat": "flat",
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
                    print(f"Available keys: {list(data.keys())}")
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

    def run_inference(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run model inference and compute Jacobians."""
        print("Running inference...")

        # Convert to torch tensors
        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            # Forward pass
            predictions = self.model(features_tensor)
            predictions = predictions.cpu().numpy().flatten()

        print("Computing Jacobians...")
        jacobians = []

        # Compute Jacobian for each sample
        for i in range(len(features)):
            sample = features_tensor[i:i + 1]  # Single sample
            sample.requires_grad_(True)

            # Forward pass
            output = self.model(sample)

            # Compute gradients
            output.backward()
            jac = sample.grad.cpu().numpy().flatten()
            jacobians.append(jac)

            # Clear gradients
            sample.grad = None

        jacobians = np.array(jacobians)

        print(f"✅ Inference complete: {len(predictions)} predictions")
        return predictions, jacobians

    def create_colormap(self, name: str):
        """Create custom colormaps for different variables."""
        if name in ["aice", "predictions"]:
            # Custom ice concentration colormap:
            # Green for open water (0-0.01), then white to blue for ice (0.01-1.0)
            from matplotlib.colors import ListedColormap
            import numpy as np

            # Create color segments
            n_total = 256
            n_open_water = int(0.01 * n_total)  # ~3 colors for 0-0.01 range
            n_ice = n_total - n_open_water      # Rest for 0.01-1.0 range

            # Open water colors (green shades)
            open_water_colors = plt.cm.Greens(np.linspace(0.3, 0.7, n_open_water))

            # Ice colors (white to blue)
            ice_colors = np.array([
                [1.0, 1.0, 1.0, 1.0],      # white
                [0.8, 0.9, 1.0, 1.0],      # very light blue
                [0.6, 0.8, 1.0, 1.0],      # light blue
                [0.4, 0.7, 1.0, 1.0],      # medium blue
                [0.2, 0.5, 0.9, 1.0],      # blue
                [0.0, 0.3, 0.8, 1.0],      # dark blue
                [0.0, 0.1, 0.6, 1.0],      # very dark blue
            ])
            ice_interp = np.zeros((n_ice, 4))
            for i in range(4):  # RGBA channels
                ice_interp[:, i] = np.interp(
                    np.linspace(0, len(ice_colors)-1, n_ice),
                    np.arange(len(ice_colors)),
                    ice_colors[:, i]
                )

            # Combine colors
            all_colors = np.vstack([open_water_colors, ice_interp])
            return ListedColormap(all_colors, name='ice_concentration')

        elif (
            "temp" in name.lower()
            or "sst" in name.lower()
            or "tsfc" in name.lower()
        ):
            # Temperature: blue to red
            return plt.cm.RdBu_r
        elif "jacobian" in name.lower():
            # Jacobian: centered around zero
            return plt.cm.RdBu
        else:
            # Default
            return plt.cm.viridis

    def _plot_ice_concentration(
        self,
        ax,
        lons: np.ndarray,
        lats: np.ndarray,
        ice_data: np.ndarray,
        targets: np.ndarray,
        title: str,
        extent: list,
        transform,
        colorbar_label: str = "Ice fraction",
        show_ice_edge: bool = True,
        use_data_range: bool = False
    ):
        """Helper function to plot ice concentration data with consistent styling."""
        # Set up map projection
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
        ax.gridlines(draw_labels=True, alpha=0.3)

        # Set color scale range
        if use_data_range:
            vmin, vmax = np.min(ice_data), np.max(ice_data)
        else:
            vmin, vmax = 0, 1

        # Main scatter plot
        scatter = ax.scatter(
            lons,
            lats,
            c=ice_data,
            s=0.5,
            cmap=self.create_colormap("aice"),
            vmin=vmin,
            vmax=vmax,
            transform=transform,
        )

        # Add ice edge overlay if requested
        if show_ice_edge:
            ice_edge_mask = (targets >= 0.14) & (targets <= 0.16)
            if np.any(ice_edge_mask):
                ax.scatter(
                    lons[ice_edge_mask],
                    lats[ice_edge_mask],
                    c='red',
                    s=1.5,
                    alpha=0.7,
                    transform=transform,
                    label='Ice Edge (~0.15)'
                )
                ax.legend(loc='upper right', fontsize=8)

        # Add title as text box inside the plot instead of as axis title
        ax.text(0.02, 0.98, title, transform=ax.transAxes,
                fontsize=10, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.colorbar(scatter, ax=ax, shrink=0.3, label=colorbar_label)
        return scatter

    def plot_domain_fields(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        jacobians: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        domain: str,
        output_dir: str = "plots",
    ):
        """Create two separate figures: ice concentrations and Jacobian components."""

        Path(output_dir).mkdir(exist_ok=True)

        # Choose projection based on domain
        if domain.lower() == "arctic":
            projection = ccrs.NorthPolarStereo()
            extent = [-180, 180, 50, 90]
        else:  # antarctic
            projection = ccrs.SouthPolarStereo()
            extent = [-180, 180, -90, -50]
        transform = ccrs.PlateCarree()

        # Calculate RMSE and correlation for titles
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        corr = np.corrcoef(predictions, targets)[0, 1]

        # ===== FIGURE 1: ICE CONCENTRATIONS =====
        fig1 = plt.figure(figsize=(16, 8))

        # 1. Target ice concentration (left)
        ax1 = plt.subplot(1, 2, 1, projection=projection)
        self._plot_ice_concentration(
            ax1, lons, lats, targets, targets,
            "Observed Ice Concentration",
            extent, transform,
            show_ice_edge=True
        )

        # 2. Predicted ice concentration (right)
        pred_min, pred_max = np.min(predictions), np.max(predictions)
        ax2 = plt.subplot(1, 2, 2, projection=projection)

        title = (f"Predicted Ice Concentration\n"
                 f"Range: [{pred_min:.3f}, {pred_max:.3f}]")

        self._plot_ice_concentration(
            ax2, lons, lats, predictions, targets,
            title, extent, transform,
            show_ice_edge=True,
            use_data_range=False
        )

        # Update legend for prediction plot
        legend = ax2.get_legend()
        if legend:
            legend.get_texts()[0].set_text('Observed Ice Edge (~0.15)')

        # Add title for ice concentration figure
        ice_title = (f"IceNet Ice Concentration - {domain.title()} Domain\n"
                    f"RMSE: {rmse:.4f} | Correlation: {corr:.4f}")
        plt.figtext(0.5, 0.95, ice_title,
                   fontsize=14, fontweight="bold",
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save ice concentration plot
        ice_output_file = f"{output_dir}/icenet_ice_concentration_{domain.lower()}.png"
        plt.savefig(ice_output_file, dpi=150, bbox_inches="tight")
        print(f"Saved ice concentration plot: {ice_output_file}")

        # ===== FIGURE 2: JACOBIAN COMPONENTS =====
        fig2 = plt.figure(figsize=(32, 24))

        # Helper function to create jacobian subplot
        def create_jacobian_subplot(subplot_idx, feature_idx, feature_name):
            if feature_idx >= 0 and feature_idx < jacobians.shape[1]:
                sensitivity = jacobians[:, feature_idx]

                # Better color scaling - use percentiles or min/max if std is too small
                std_val = np.std(sensitivity)
                min_val, max_val = np.min(sensitivity), np.max(sensitivity)

                # Use 95th percentile range for better visualization
                p5, p95 = np.percentile(sensitivity, [5, 95])

                # Choose appropriate range
                if std_val > 1e-8:  # If std is reasonable, use 2*std
                    vmin, vmax = -2*std_val, 2*std_val
                elif max_val - min_val > 1e-8:  # If range is reasonable, use percentiles
                    vmin, vmax = p5, p95
                else:  # For very small values, use actual range
                    vmin, vmax = min_val, max_val

                ax = plt.subplot(3, 6, subplot_idx, projection=projection)
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, alpha=0.5)
                ax.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
                ax.gridlines(draw_labels=True, alpha=0.3)

                # Debug info for troubleshooting
                print(f"Jacobian {feature_name}: range=[{min_val:.6f}, {max_val:.6f}], "
                      f"std={std_val:.6f}, vmin={vmin:.6f}, vmax={vmax:.6f}")

                scatter = ax.scatter(
                    lons, lats, c=sensitivity, s=1.0,  # Increased point size
                    cmap=self.create_colormap("jacobian"),
                    transform=transform, vmin=vmin, vmax=vmax, alpha=0.7
                )

                # Add title as text box inside the plot
                title_text = (f"dice/d{feature_name.lower()}\n"
                             f"range: [{min_val:.2e}, {max_val:.2e}]")
                ax.text(0.02, 0.98, title_text, transform=ax.transAxes,
                       fontsize=9, fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.8))

                plt.colorbar(scatter, ax=ax, shrink=0.3,
                           label=f"dice/d{feature_name.lower()}")
            else:
                # Show placeholder if feature not available
                ax = plt.subplot(3, 6, subplot_idx, projection=projection)
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, alpha=0.5)
                ax.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
                ax.gridlines(draw_labels=True, alpha=0.3)
                # Add title as text box instead of axis title
                ax.text(0.02, 0.98, f"{feature_name} not available",
                       transform=ax.transAxes, fontsize=10,
                       fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.8))
                ax.text(0.5, 0.5, f'No {feature_name}',
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat',
                               alpha=0.8))

        # Create Jacobian subplots for all 17 features in a 3x6 layout
        for i, feature_name in enumerate(self.feature_names):
            subplot_idx = i + 1  # subplot indices start from 1
            create_jacobian_subplot(subplot_idx, i, feature_name)

        # Fill remaining slots with empty plots if needed (3x6 = 18 slots, we have 17 features)
        if len(self.feature_names) < 18:
            for empty_idx in range(len(self.feature_names) + 1, 19):
                ax = plt.subplot(3, 6, empty_idx, projection=projection)
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, alpha=0.5)
                ax.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
                ax.gridlines(draw_labels=True, alpha=0.3)
                ax.text(0.5, 0.5, 'Empty', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightgray',
                               alpha=0.5))

        # Add title for Jacobian figure
        jac_title = f"IceNet Jacobian Sensitivity - {domain.title()} Domain"
        plt.figtext(0.5, 0.95, jac_title,
                   fontsize=14, fontweight="bold",
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save Jacobian plot
        jac_output_file = f"{output_dir}/icenet_jacobian_{domain.lower()}.png"
        plt.savefig(jac_output_file, dpi=150, bbox_inches="tight")
        print(f"Saved Jacobian plot: {jac_output_file}")

        return fig1, fig2

    def create_summary_statistics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        jacobians: np.ndarray,
        features: np.ndarray,
        domain: str,
    ):
        """Print simplified summary statistics."""
        print(f"\n📊 {domain.title()} Domain Statistics:")
        print("=" * 40)

        # Prediction metrics
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mae = np.mean(np.abs(predictions - targets))
        corr = np.corrcoef(predictions, targets)[0, 1]

        print(f"Prediction RMSE: {rmse:.4f}")
        print(f"Prediction MAE:  {mae:.4f}")
        print(f"Correlation:     {corr:.4f}")

        # Ice edge statistics
        ice_edge_mask = (targets >= 0.10) & (targets <= 0.16)
        n_ice_edge = np.sum(ice_edge_mask)
        total_points = len(targets)
        ice_edge_percentage = n_ice_edge / total_points * 100

        print(f"\nIce Edge Statistics (0.10-0.16 concentration):")
        print(f"  Points: {n_ice_edge}/{total_points} ({ice_edge_percentage:.1f}%)")

        if n_ice_edge > 0:
            ice_edge_pred = predictions[ice_edge_mask]
            ice_edge_targets = targets[ice_edge_mask]
            ice_edge_rmse = np.sqrt(np.mean((ice_edge_pred - ice_edge_targets) ** 2))
            ice_edge_bias = np.mean(ice_edge_pred - ice_edge_targets)

            print(f"  Ice Edge RMSE: {ice_edge_rmse:.4f}")
            print(f"  Ice Edge Bias: {ice_edge_bias:.4f} (pred - obs)")
            print(f"  Predicted range at ice edge: [{np.min(ice_edge_pred):.3f}, {np.max(ice_edge_pred):.3f}]")

        # Open water statistics
        open_water_mask = targets < 0.05
        n_open_water = np.sum(open_water_mask)
        open_water_percentage = n_open_water / total_points * 100

        print(f"\nOpen Water Statistics (<0.05 concentration):")
        print(f"  Points: {n_open_water}/{total_points} ({open_water_percentage:.1f}%)")

        if n_open_water > 0:
            open_water_pred = predictions[open_water_mask]
            open_water_targets = targets[open_water_mask]
            open_water_rmse = np.sqrt(np.mean((open_water_pred - open_water_targets) ** 2))
            open_water_bias = np.mean(open_water_pred - open_water_targets)

            print(f"  Open Water RMSE: {open_water_rmse:.4f}")
            print(f"  Open Water Bias: {open_water_bias:.4f} (pred - obs)")
            print(f"  Predicted range in open water: [{np.min(open_water_pred):.3f}, {np.max(open_water_pred):.3f}]")

        # Jacobian statistics for each feature
        for i, feature_name in enumerate(self.feature_names):
            jacobian_vals = jacobians[:, i]
            print(f"\n{feature_name} Jacobian dice/d{feature_name.lower()}:")
            print(f"  Range: [{np.min(jacobian_vals):.6f}, "
                  f"{np.max(jacobian_vals):.6f}]")
            print(f"  Mean: {np.mean(jacobian_vals):.6f}")
            print(f"  Std:  {np.std(jacobian_vals):.6f}")

            # Input statistics for this feature
            feature_vals = features[:, i]
            unit = self.feature_units[i] if i < len(self.feature_units) else ""
            print(f"\n{feature_name} Input Statistics:")
            print(f"  Range: [{np.min(feature_vals):.2f}, "
                  f"{np.max(feature_vals):.2f}]{unit}")
            print(f"  Mean: {np.mean(feature_vals):.2f}{unit}")
            print(f"  Std:  {np.std(feature_vals):.2f}{unit}")

        # Basic feature sensitivity ranking
        jac_magnitudes = np.abs(jacobians)
        most_sensitive = np.argmax(jac_magnitudes, axis=1)

        print("\nMost sensitive features:")
        for i, feature in enumerate(self.feature_names):
            count = np.sum(most_sensitive == i)
            percentage = count / len(most_sensitive) * 100
            if percentage > 1.0:  # Only show features with >1% sensitivity
                print(f"  {feature}: {percentage:.1f}% of points")


def thin_data(inputs, targets, lons, lats, fraction):
    if fraction < 1.0:
        n = len(targets)
        n_thin = int(n * fraction)
        idx = np.random.choice(n, n_thin, replace=False)
        return inputs[idx], targets[idx], lons[idx], lats[idx]
    return inputs, targets, lons, lats


def thin_all(features, targets, lons, lats, predictions, jacobians, fraction):
    if fraction < 1.0:
        n = len(targets)
        n_thin = int(n * fraction)
        idx = np.random.choice(n, n_thin, replace=False)
        return (features[idx], targets[idx], lons[idx], lats[idx], predictions[idx], jacobians[idx])
    return features, targets, lons, lats, predictions, jacobians


def main():
    parser = argparse.ArgumentParser(
        description="IceNet Inference and Jacobian Visualization"
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
    parser.add_argument('--thin-fraction', type=float, default=1.0,
                        help='Fraction of data to plot (e.g. 0.5 for 50 percent)')

    args = parser.parse_args()

    # Initialize plotter
    plotter = IceNetInferencePlotter(args.model, args.config)

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
            # Filter data for domain using same criteria as training
            # Default to clean_data=True to match training behavior
            clean_data = True
            features, targets, lons, lats, mask, shape = plotter.filter_domain(
                data, domain, clean_data
            )

            if len(features) == 0:
                print(f"⚠️ No valid data points found for {domain} domain")
                continue

            # Run inference and compute Jacobians
            predictions, jacobians = plotter.run_inference(features)

            # Thin all arrays together
            features, targets, lons, lats, predictions, jacobians = thin_all(
                features, targets, lons, lats, predictions, jacobians, args.thin_fraction)

            # Create plots
            fig1, fig2 = plotter.plot_domain_fields(
                features,
                targets,
                predictions,
                jacobians,
                lons,
                lats,
                domain,
                args.output_dir,
            )

            # Print statistics
            plotter.create_summary_statistics(
                predictions, targets, jacobians, features, domain
            )

            plt.close(fig1)  # Free memory
            plt.close(fig2)  # Free memory

        except Exception as e:
            print(f"❌ Error processing {domain} domain: {e}")
            continue

    print(f"\n✅ Analysis complete! Check {args.output_dir}/ for plots")


if __name__ == "__main__":
    main()
