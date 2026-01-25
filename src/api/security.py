"""
Security middleware and utilities.
Implements authentication, rate limiting, and security headers.
"""
from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from functools import wraps
import logging
from typing import Callable

from config import settings

logger = logging.getLogger(__name__)

# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header.
    
    Args:
        api_key: API key from header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key is None:
        logger.warning("Missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key != settings.api_key:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


# Rate Limiting
limiter = Limiter(key_func=get_remote_address)


def setup_security_headers(app):
    """
    Add security headers to all responses.
    
    Args:
        app: FastAPI application instance
    """
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net data:"
        
        return response


def log_request(func: Callable) -> Callable:
    """
    Decorator to log API requests.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with logging
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"API request: {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"API request successful: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"API request failed: {func.__name__}, Error: {str(e)}")
            raise
    
    return wrapper
