"""
Configuration module for the Sentiment Analysis API.
Implements configuration management following best practices.
"""
from typing import List, Literal, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    Application settings with validation and type hints.
    Uses environment variables with fallback defaults.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Configuration
    api_key: str = Field(default="dev-key-change-in-production", description="API authentication key")
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port number")
    api_reload: bool = Field(default=False, description="Enable auto-reload in development")
    environment: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Application environment"
    )
    
    # Model Configuration
    model_type: Literal["baseline", "distilbert"] = Field(
        default="distilbert",
        description="Type of model to use"
    )
    model_path: str = Field(default="models/best_model.pkl", description="Path to trained model")
    max_length: int = Field(default=128, description="Maximum token length for text input")
    batch_size: int = Field(default=32, description="Batch size for model inference")
    
    # Data Configuration
    data_path: str = Field(
        default="../yelp_academic_dataset_review.json",
        description="Path to dataset"
    )
    sample_size: int = Field(default=100000, description="Number of samples to use from dataset")
    test_size: float = Field(default=0.2, description="Proportion of data for testing")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = Field(
        default="sentiment-analysis",
        description="MLflow experiment name"
    )
    
    # Security Configuration
    allowed_origins: Union[str, List[str]] = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="CORS allowed origins (comma-separated string or list)"
    )
    rate_limit: str = Field(default="100/minute", description="API rate limit")
    
    @validator("allowed_origins", pre=True, always=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list."""
        if v is None or v == "":
            return ["http://localhost:3000", "http://localhost:8000"]
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return v
        return ["http://localhost:3000", "http://localhost:8000"]
    
    @validator("test_size")
    def validate_test_size(cls, v):
        """Ensure test size is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("test_size must be between 0 and 1")
        return v
    
    @validator("sample_size")
    def validate_sample_size(cls, v):
        """Ensure sample size is positive."""
        if v <= 0:
            raise ValueError("sample_size must be positive")
        return v


# Singleton instance
settings = Settings()
