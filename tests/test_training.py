"""
Integration tests for training functionality
"""

import torch
import yaml
import numpy as np

from icenet.training import IceNetTrainer, load_config, create_default_config


class TestTraining:
    """Test cases for training functionality."""

    def test_config_loading(self, temp_dir):
        """Test configuration loading from YAML file."""
        config_data = {
            'model': {'input_size': 7, 'hidden_size': 16, 'output_size': 1},
            'training': {'epochs': 1, 'batch_size': 4},
            'data': {'validation_split': 0.2, 'num_workers': 0}
        }

        config_path = temp_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        loaded_config = load_config(str(config_path))
        assert loaded_config['model']['input_size'] == 7
        assert loaded_config['training']['epochs'] == 1

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = create_default_config()

        # Check required sections exist
        assert 'model' in config
        assert 'training' in config
        assert 'data' in config

        # Check some default values
        assert config['model']['input_size'] == 7
        assert config['model']['hidden_size'] == 16
        assert config['model']['output_size'] == 1

    def test_trainer_initialization(self, sample_config, temp_dir):
        """Test trainer initialization."""
        # Modify config for testing
        sample_config['data']['data_path'] = str(temp_dir / "test_data.npz")

        trainer = IceNetTrainer(sample_config)

        assert trainer.config == sample_config
        assert trainer.device is not None
        assert trainer.model is not None

    def test_data_creation_and_loading(
            self, sample_config, sample_data, temp_dir):
        """Test data creation and loading."""
        X, y = sample_data

        # Save sample data in numpy format
        data_path = temp_dir / "test_data.npz"
        np.savez(
            data_path,
            inputs=X.numpy(),
            targets=y.numpy(),
            input_mean=X.mean(dim=0).numpy(),
            input_std=X.std(dim=0).numpy()
        )

        # Update config
        sample_config['data']['data_path'] = str(data_path)
        sample_config['data']['validation_split'] = 0.3

        trainer = IceNetTrainer(sample_config)
        train_loader, val_loader = trainer.load_data(str(data_path))

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0

    def test_training_step(self, sample_config, sample_data, temp_dir):
        """Test a single training step."""
        X, y = sample_data

        # Save sample data in numpy format
        data_path = temp_dir / "test_data.npz"
        np.savez(
            data_path,
            inputs=X.numpy(),
            targets=y.numpy(),
            input_mean=X.mean(dim=0).numpy(),
            input_std=X.std(dim=0).numpy()
        )

        # Configure for quick test
        sample_config['data']['data_path'] = str(data_path)
        sample_config['training']['epochs'] = 1
        sample_config['training']['batch_size'] = 10

        trainer = IceNetTrainer(sample_config)
        train_loader, val_loader = trainer.load_data(str(data_path))

        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader)

        # Check that training ran
        assert train_loss > 0
        # train() adds to history, train_epoch() doesn't
        assert len(trainer.history['train_loss']) == 0

    def test_validation_step(self, sample_config, sample_data, temp_dir):
        """Test validation step."""
        X, y = sample_data

        # Save sample data in numpy format
        data_path = temp_dir / "test_data.npz"
        np.savez(
            data_path,
            inputs=X.numpy(),
            targets=y.numpy(),
            input_mean=X.mean(dim=0).numpy(),
            input_std=X.std(dim=0).numpy()
        )

        sample_config['data']['data_path'] = str(data_path)

        trainer = IceNetTrainer(sample_config)
        train_loader, val_loader = trainer.load_data(str(data_path))

        val_loss = trainer.validate(val_loader)

        assert val_loss > 0
        assert isinstance(val_loss, float)

    def test_model_saving(self, sample_config, sample_data, temp_dir):
        """Test model saving functionality."""
        X, y = sample_data

        # Save sample data in numpy format
        data_path = temp_dir / "test_data.npz"
        np.savez(
            data_path,
            inputs=X.numpy(),
            targets=y.numpy(),
            input_mean=X.mean(dim=0).numpy(),
            input_std=X.std(dim=0).numpy()
        )

        sample_config['data']['data_path'] = str(data_path)

        trainer = IceNetTrainer(sample_config)

        # Save model
        model_path = temp_dir / "test_model.pth"
        trainer.save_model(str(model_path))

        assert model_path.exists()

        # Check if normalization file was also created
        norm_files = list(temp_dir.glob("normalization.*"))
        assert len(norm_files) > 0

    def test_optimizer_creation(self, sample_config):
        """Test optimizer creation with different types."""
        # Test Adam optimizer
        sample_config['training']['optimizer']['type'] = 'adam'
        trainer = IceNetTrainer(sample_config)
        assert isinstance(trainer.optimizer, torch.optim.Adam)

        # Test SGD optimizer
        sample_config['training']['optimizer']['type'] = 'sgd'
        trainer = IceNetTrainer(sample_config)
        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_scheduler_creation(self, sample_config):
        """Test learning rate scheduler creation."""
        # Test StepLR scheduler
        sample_config['training']['scheduler']['type'] = 'step'
        trainer = IceNetTrainer(sample_config)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

        # Test ReduceLROnPlateau scheduler
        sample_config['training']['scheduler']['type'] = 'plateau'
        trainer = IceNetTrainer(sample_config)
        scheduler_type = torch.optim.lr_scheduler.ReduceLROnPlateau
        assert isinstance(trainer.scheduler, scheduler_type)
