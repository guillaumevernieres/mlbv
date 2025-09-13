"""
Tests for data preparation functionality
"""

import numpy as np

from icenet.data import IceDataPreparer


class TestDataPreparation:
    """Test cases for data preparation functionality."""

    def test_data_preparer_initialization(self):
        """Test DataPreparer initialization."""
        config = {
            'domain': {
                'pole': 'north',
                'clean_data': True
            }
        }

        preparer = IceDataPreparer(config)

        assert preparer.pole == 'north'
        assert preparer.clean_data is True
        assert preparer.config == config

    def test_data_preparer_south_pole(self):
        """Test DataPreparer with south pole configuration."""
        config = {
            'domain': {
                'pole': 'south',
                'clean_data': False
            }
        }

        preparer = IceDataPreparer(config)

        assert preparer.pole == 'south'
        assert preparer.clean_data is False

    def test_normalization_stats_computation(self):
        """Test computation of normalization statistics."""
        config = {'domain': {'pole': 'north', 'clean_data': True}}
        preparer = IceDataPreparer(config)

        # Create sample data
        patterns = np.random.randn(100, 7)

        mean, std = preparer.compute_normalization_stats(patterns)

        assert mean.shape == (7,)
        assert std.shape == (7,)
        assert np.allclose(mean, patterns.mean(axis=0), rtol=1e-5)
        assert np.allclose(std, patterns.std(axis=0), rtol=1e-5)

    def test_variable_names_mapping(self):
        """Test that variable names are properly mapped."""
        config = {'domain': {'pole': 'north', 'clean_data': True}}
        preparer = IceDataPreparer(config)

        expected_vars = {
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

        assert preparer.var_names == expected_vars

    def test_data_filtering_parameters(self):
        """Test data filtering with different parameters."""
        # Test different configurations
        configs = [
            {'domain': {'pole': 'north', 'clean_data': True}},
            {'domain': {'pole': 'south', 'clean_data': True}},
            {'domain': {'pole': 'north', 'clean_data': False}},
        ]

        for config in configs:
            preparer = IceDataPreparer(config)
            assert preparer.config == config
