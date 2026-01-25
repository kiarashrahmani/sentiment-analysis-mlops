"""
API schemas for request/response validation.
Uses Pydantic for automatic validation and documentation.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Literal
from datetime import datetime


class PredictionRequest(BaseModel):
    """Schema for single prediction request."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to classify for sentiment"
    )
    
    @validator('text')
    def text_not_empty(cls, v):
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This restaurant has amazing food and great service!"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of texts to classify"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Ensure all texts are valid."""
        for text in v:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("All texts must be non-empty strings")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "The food was excellent!",
                    "Terrible experience, would not recommend.",
                    "It was okay, nothing special."
                ]
            }
        }


class SentimentProbabilities(BaseModel):
    """Schema for sentiment probabilities."""
    
    negative: float = Field(..., ge=0.0, le=1.0, description="Probability of negative sentiment")
    neutral: float = Field(..., ge=0.0, le=1.0, description="Probability of neutral sentiment")
    positive: float = Field(..., ge=0.0, le=1.0, description="Probability of positive sentiment")
    
    class Config:
        schema_extra = {
            "example": {
                "negative": 0.05,
                "neutral": 0.15,
                "positive": 0.80
            }
        }


class PredictionResponse(BaseModel):
    """Schema for single prediction response."""
    
    text: str = Field(..., description="Input text")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of prediction")
    probabilities: SentimentProbabilities = Field(..., description="Probabilities for all classes")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This restaurant has amazing food and great service!",
                "sentiment": "positive",
                "confidence": 0.95,
                "probabilities": {
                    "negative": 0.02,
                    "neutral": 0.03,
                    "positive": 0.95
                },
                "timestamp": "2024-01-25T10:30:00Z"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch processing timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "text": "The food was excellent!",
                        "sentiment": "positive",
                        "confidence": 0.92,
                        "probabilities": {
                            "negative": 0.03,
                            "neutral": 0.05,
                            "positive": 0.92
                        },
                        "timestamp": "2024-01-25T10:30:00Z"
                    }
                ],
                "total_count": 1,
                "timestamp": "2024-01-25T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: str = Field(..., description="Type of model loaded")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_type": "distilbert",
                "version": "1.0.0",
                "timestamp": "2024-01-25T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input text",
                "detail": "Text cannot be empty",
                "timestamp": "2024-01-25T10:30:00Z"
            }
        }
