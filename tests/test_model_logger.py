"""
Unit tests for the model_logger module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import mlflow
from mlflow.exceptions import MlflowException
from src.utils.model_logger import ModelLogger, log_model_simple


class TestModelLogger(unittest.TestCase):
    """Test cases for ModelLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = ModelLogger(registry_uri="databricks-uc")
    
    @patch('mlflow.pyfunc.log_model')
    @patch('mlflow.MlflowClient')
    def test_log_model_creates_new_version(self, mock_client_class, mock_log_model):
        """Test that logging a model with existing registered name creates new version."""
        # Setup mocks
        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/123/model"
        mock_log_model.return_value = mock_model_info
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock existing model with versions
        mock_registered_model = Mock()
        mock_registered_model.name = "test_model"
        mock_client.get_registered_model.return_value = mock_registered_model
        
        mock_version = Mock()
        mock_version.version = "2"
        mock_client.get_latest_versions.return_value = [mock_version]
        
        # Test logging
        result = self.logger.log_model(
            model="test_model.py",
            registered_model_name="test_model"
        )
        
        # Assertions
        self.assertEqual(result.model_uri, "runs:/123/model")
        mock_log_model.assert_called_once()
        mock_client.get_registered_model.assert_called_once_with("test_model")
        mock_client.get_latest_versions.assert_called_once()
    
    @patch('mlflow.pyfunc.log_model')
    @patch('mlflow.MlflowClient')
    def test_log_model_creates_new_model(self, mock_client_class, mock_log_model):
        """Test that logging a model with new registered name creates first version."""
        # Setup mocks
        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/456/model"
        mock_log_model.return_value = mock_model_info
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock model not found
        mock_client.get_registered_model.side_effect = MlflowException("RESOURCE_DOES_NOT_EXIST")
        
        # Test logging
        result = self.logger.log_model(
            model="new_model.py",
            registered_model_name="new_model"
        )
        
        # Assertions
        self.assertEqual(result.model_uri, "runs:/456/model")
        mock_log_model.assert_called_once()
        mock_client.get_registered_model.assert_called_once_with("new_model")
    
    @patch('mlflow.pyfunc.log_model')
    def test_log_model_without_registration(self, mock_log_model):
        """Test that logging without registered_model_name works."""
        # Setup mocks
        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/789/model"
        mock_log_model.return_value = mock_model_info
        
        # Test logging without registration
        result = self.logger.log_model(
            model="unregistered_model.py"
        )
        
        # Assertions
        self.assertEqual(result.model_uri, "runs:/789/model")
        mock_log_model.assert_called_once()
    
    @patch('mlflow.MlflowClient')
    def test_get_model_version_info(self, mock_client_class):
        """Test retrieving model version information."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_model = Mock()
        mock_model.name = "test_model"
        mock_model.creation_timestamp = 1234567890
        mock_model.last_updated_timestamp = 1234567900
        mock_model.description = "Test model"
        mock_client.get_registered_model.return_value = mock_model
        
        mock_version1 = Mock()
        mock_version1.version = "1"
        mock_version1.current_stage = "Production"
        mock_version1.status = "READY"
        mock_version1.creation_timestamp = 1234567890
        mock_version1.run_id = "run1"
        
        mock_version2 = Mock()
        mock_version2.version = "2"
        mock_version2.current_stage = "None"
        mock_version2.status = "READY"
        mock_version2.creation_timestamp = 1234567900
        mock_version2.run_id = "run2"
        
        mock_client.search_model_versions.return_value = [mock_version1, mock_version2]
        
        # Test getting version info
        info = self.logger.get_model_version_info("test_model")
        
        # Assertions
        self.assertEqual(info["name"], "test_model")
        self.assertEqual(info["total_versions"], 2)
        self.assertEqual(len(info["versions"]), 2)
        self.assertEqual(info["versions"][0]["version"], "1")
        self.assertEqual(info["versions"][0]["stage"], "Production")
        self.assertEqual(info["versions"][1]["version"], "2")
    
    @patch('mlflow.MlflowClient')
    def test_get_model_version_info_not_found(self, mock_client_class):
        """Test retrieving version info for non-existent model."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_registered_model.side_effect = MlflowException("RESOURCE_DOES_NOT_EXIST")
        
        # Test getting version info for non-existent model
        info = self.logger.get_model_version_info("nonexistent_model")
        
        # Assertions
        self.assertIn("error", info)
        self.assertEqual(info["error"], "Model 'nonexistent_model' not found")


class TestLogModelSimple(unittest.TestCase):
    """Test cases for log_model_simple function."""
    
    @patch('mlflow.start_run')
    @patch('mlflow.set_experiment')
    @patch('src.utils.model_logger.ModelLogger.log_model')
    def test_log_model_simple(self, mock_log_model, mock_set_experiment, mock_start_run):
        """Test the simplified log_model_simple function."""
        # Setup mocks
        mock_context = MagicMock()
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/abc/model"
        mock_log_model.return_value = mock_model_info
        
        # Test simple logging
        result = log_model_simple(
            model="simple_model.py",
            registered_model_name="simple_model",
            experiment_name="/Users/test/experiment",
            run_name="test_run"
        )
        
        # Assertions
        mock_set_experiment.assert_called_once_with("/Users/test/experiment")
        mock_start_run.assert_called_once_with(run_name="test_run")
        mock_log_model.assert_called_once()
        self.assertEqual(result.model_uri, "runs:/abc/model")


if __name__ == '__main__':
    unittest.main()