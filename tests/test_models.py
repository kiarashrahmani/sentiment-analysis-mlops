"""Tests for model implementations."""
import pytest
import numpy as np
from pathlib import Path

from src.models import BaselineModel, ModelEvaluator


class TestBaselineModel:
    """Test cases for BaselineModel."""
    
    def test_model_training(self):
        """Test model training."""
        model = BaselineModel()
        
        X_train = [
            "This is great!",
            "Terrible experience",
            "It's okay",
            "Amazing product",
            "Very disappointed"
        ]
        y_train = ["positive", "negative", "neutral", "positive", "negative"]
        
        model.train(X_train, y_train)
        
        assert model.is_trained
    
    def test_model_prediction(self):
        """Test single prediction."""
        model = BaselineModel()
        
        X_train = [
            "This is great!",
            "Terrible experience",
            "It's okay",
            "Amazing product",
            "Very disappointed"
        ]
        y_train = ["positive", "negative", "neutral", "positive", "negative"]
        
        model.train(X_train, y_train)
        
        prediction = model.predict("This is wonderful!")
        
        assert prediction in ["positive", "negative", "neutral"]
    
    def test_model_batch_prediction(self):
        """Test batch prediction."""
        model = BaselineModel()
        
        X_train = [
            "This is great!",
            "Terrible experience",
            "It's okay",
            "Amazing product",
            "Very disappointed"
        ]
        y_train = ["positive", "negative", "neutral", "positive", "negative"]
        
        model.train(X_train, y_train)
        
        predictions = model.predict(["Great!", "Bad!"])
        
        assert len(predictions) == 2
        assert all(p in ["positive", "negative", "neutral"] for p in predictions)
    
    def test_model_predict_proba(self):
        """Test probability predictions."""
        model = BaselineModel()
        
        X_train = [
            "This is great!",
            "Terrible experience",
            "It's okay",
            "Amazing product",
            "Very disappointed"
        ]
        y_train = ["positive", "negative", "neutral", "positive", "negative"]
        
        model.train(X_train, y_train)
        
        proba = model.predict_proba("This is wonderful!")
        
        assert isinstance(proba, dict)
        assert set(proba.keys()) == {"positive", "negative", "neutral"}
        assert abs(sum(proba.values()) - 1.0) < 0.01  # Sum should be ~1.0
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        model = BaselineModel()
        
        with pytest.raises(RuntimeError):
            model.predict("Some text")
    
    def test_model_save_load(self, temp_model_dir):
        """Test model saving and loading."""
        model = BaselineModel()
        
        X_train = [
            "This is great!",
            "Terrible experience",
            "It's okay",
            "Amazing product",
            "Very disappointed"
        ]
        y_train = ["positive", "negative", "neutral", "positive", "negative"]
        
        model.train(X_train, y_train)
        
        # Save model
        model_path = Path(temp_model_dir) / "test_model.pkl"
        model.save(str(model_path))
        
        # Load model
        loaded_model = BaselineModel()
        loaded_model.load(str(model_path))
        
        # Test loaded model
        prediction = loaded_model.predict("This is wonderful!")
        assert prediction in ["positive", "negative", "neutral"]


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""
    
    def test_evaluate_metrics(self):
        """Test evaluation metrics calculation."""
        y_true = ["positive", "negative", "neutral", "positive", "negative"]
        y_pred = ["positive", "negative", "neutral", "positive", "positive"]
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0
