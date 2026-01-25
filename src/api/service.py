"""
Model service for handling predictions.
Implements business logic separation from API layer.
"""
from typing import Dict, List, Union
import logging
from pathlib import Path

from src.models import ModelInterface, BaselineModel, DistilBERTModel
from config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service class for model operations.
    Single Responsibility: Manages model loading and predictions.
    """
    
    def __init__(self):
        """Initialize model service."""
        self.model: ModelInterface = None
        self.model_type: str = None
    
    def load_model(self, model_path: str = None, model_type: str = None) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to model file
            model_type: Type of model to load
        """
        model_path = model_path or settings.model_path
        model_type = model_type or settings.model_type
        
        logger.info(f"Loading {model_type} model from {model_path}")
        
        try:
            # Create model instance
            if model_type == "baseline":
                self.model = BaselineModel()
            elif model_type == "distilbert":
                self.model = DistilBERTModel(
                    max_length=settings.max_length,
                    batch_size=settings.batch_size
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load model weights
            if Path(model_path).exists():
                self.model.load(model_path)
                self.model_type = model_type
                logger.info(f"Model loaded successfully: {model_type}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Make prediction for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        sentiment = self.model.predict(text)
        probabilities = self.model.predict_proba(text)
        confidence = max(probabilities.values())
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probabilities
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Make predictions for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        sentiments = self.model.predict(texts)
        probabilities_list = self.model.predict_proba(texts)
        
        results = []
        for text, sentiment, probabilities in zip(texts, sentiments, probabilities_list):
            confidence = max(probabilities.values())
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": probabilities
            })
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.model.is_trained
    
    def get_model_type(self) -> str:
        """Get the type of loaded model."""
        return self.model_type if self.model_type else "none"


# Singleton instance
model_service = ModelService()
