"""
Model interfaces and base classes following SOLID principles.
Liskov Substitution Principle: All models can be used interchangeably.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """
    Abstract base class for all sentiment models.
    Defines the contract that all models must follow.
    """
    
    @abstractmethod
    def train(self, X_train: Union[List[str], pd.Series], y_train: Union[List[str], pd.Series]) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """Make predictions on text(s)."""
        pass
    
    @abstractmethod
    def predict_proba(self, texts: Union[str, List[str]]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: Union[List[str], pd.Series], y_test: Union[List[str], pd.Series]) -> Dict[str, Any]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass


class BaseModel(ModelInterface):
    """
    Base model class with common functionality.
    Template Method Pattern: Defines common structure for all models.
    """
    
    def __init__(self):
        """Initialize base model."""
        self.model = None
        self.classes_ = ['negative', 'neutral', 'positive']
        self.is_trained = False
    
    def _validate_trained(self) -> None:
        """Ensure model is trained before prediction."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
    
    def _ensure_list(self, texts: Union[str, List[str]]) -> List[str]:
        """Convert single text to list for uniform processing."""
        if isinstance(texts, str):
            return [texts]
        return list(texts)
    
    def _format_probabilities(
        self,
        proba_array: np.ndarray,
        single_input: bool
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Format probability arrays into dictionaries.
        
        Args:
            proba_array: Numpy array of probabilities
            single_input: Whether input was a single text
            
        Returns:
            Dictionary or list of dictionaries with class probabilities
        """
        result = []
        for proba in proba_array:
            result.append({
                class_name: float(prob)
                for class_name, prob in zip(self.classes_, proba)
            })
        
        return result[0] if single_input else result
    
    def save(self, path: str) -> None:
        """
        Save model to disk using joblib.
        
        Args:
            path: File path to save model
        """
        self._validate_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'classes': self.classes_,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.classes_ = model_data['classes']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {path}")


class ModelEvaluator:
    """
    Utility class for model evaluation.
    Single Responsibility: Only handles evaluation metrics.
    """
    
    @staticmethod
    def evaluate(
        y_true: Union[List[str], pd.Series],
        y_pred: Union[List[str], pd.Series]
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
            classification_report
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
