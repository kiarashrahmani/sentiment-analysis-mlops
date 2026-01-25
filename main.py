"""
FastAPI application for Sentiment Analysis.
Implements REST API with proper error handling, validation, and security.
"""
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import logging
from typing import Dict

from src.api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    SentimentProbabilities
)
from src.api.security import (
    verify_api_key,
    setup_security_headers,
    limiter,
    log_request
)
from src.api.service import model_service
from config import settings
from src import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Sentiment Analysis API")
    try:
        model_service.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Continue without model - health check will report unhealthy
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiment Analysis API")


# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready API for sentiment classification using ML models",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup rate limiting
app.state.limiter = limiter

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Setup security headers
setup_security_headers(app)


# Exception handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "RateLimitExceeded",
            "message": "Too many requests. Please try again later.",
            "detail": str(exc)
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "detail": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An internal error occurred",
            "detail": str(exc) if settings.environment == "development" else None
        }
    )


# API Endpoints
@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Sentiment Analysis API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API and model health status"
)
@limiter.limit(settings.rate_limit)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.
    Returns API status and model availability.
    """
    is_healthy = model_service.is_loaded()
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        model_type=model_service.get_model_type(),
        version=__version__
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict sentiment",
    description="Classify a single text input for sentiment",
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit(settings.rate_limit)
@log_request
async def predict(
    request: Request,
    prediction_request: PredictionRequest
) -> PredictionResponse:
    """
    Predict sentiment for a single text.
    
    Args:
        prediction_request: Request with text to classify
        
    Returns:
        Prediction response with sentiment and probabilities
    """
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        result = model_service.predict_single(prediction_request.text)
        
        return PredictionResponse(
            text=result["text"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=SentimentProbabilities(**result["probabilities"])
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch predict sentiment",
    description="Classify multiple texts for sentiment",
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("20/minute")  # Lower rate limit for batch
@log_request
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest
) -> BatchPredictionResponse:
    """
    Predict sentiment for multiple texts.
    
    Args:
        batch_request: Request with list of texts to classify
        
    Returns:
        Batch prediction response with list of predictions
    """
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        results = model_service.predict_batch(batch_request.texts)
        
        predictions = [
            PredictionResponse(
                text=result["text"],
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                probabilities=SentimentProbabilities(**result["probabilities"])
            )
            for result in results
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level="info"
    )
