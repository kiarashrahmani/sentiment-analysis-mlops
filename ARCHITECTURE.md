# Project Architecture & Design

## Overview

This document explains the architecture, design patterns, and best practices implemented in the Sentiment Analysis API.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│  (Web Apps, Mobile Apps, CLI Tools, Other Services)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FastAPI Application (main.py)                       │  │
│  │  • CORS Middleware                                   │  │
│  │  • Security Headers                                  │  │
│  │  • Rate Limiting                                     │  │
│  │  • Error Handling                                    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SECURITY LAYER (src/api)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • API Key Authentication                            │  │
│  │  • Request Validation (Pydantic)                     │  │
│  │  • Input Sanitization                                │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SERVICE LAYER (src/api)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Model Service                                       │  │
│  │  • Model Loading                                     │  │
│  │  • Prediction Orchestration                          │  │
│  │  • Result Formatting                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL LAYER (src/models)                  │
│  ┌──────────────┐              ┌─────────────────────────┐ │
│  │   Baseline   │              │     DistilBERT          │ │
│  │    Model     │              │       Model             │ │
│  │  TF-IDF +    │              │   Transformer-based     │ │
│  │  Logistic    │              │   Fine-tuned BERT       │ │
│  │  Regression  │              │                         │ │
│  └──────────────┘              └─────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA LAYER (src/data)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Data Loading (Chunked for large files)           │  │
│  │  • Text Cleaning                                     │  │
│  │  • Train/Test Splitting                              │  │
│  │  • Data Validation                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## SOLID Principles Implementation

### 1. Single Responsibility Principle (SRP)

Each class has one reason to change:

- **DataLoader**: Only loads data
- **TextCleaner**: Only cleans text
- **DataProcessor**: Coordinates data operations
- **Model Classes**: Handle ML operations
- **ModelService**: Manages model lifecycle
- **Security Module**: Handles authentication

**Example**:
```python
class BasicTextCleaner(TextCleanerInterface):
    """Only responsible for text cleaning."""
    def clean(self, text: str) -> str:
        # Text cleaning logic only
        pass
```

### 2. Open/Closed Principle (OCP)

Open for extension, closed for modification:

- **Abstract Interfaces**: Easy to add new implementations
- **New Models**: Add without modifying existing code
- **New Data Sources**: Implement DataLoaderInterface

**Example**:
```python
class DataLoaderInterface(ABC):
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        pass

# Easy to add new loaders without modifying existing code
class CSVLoader(DataLoaderInterface):
    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
```

### 3. Liskov Substitution Principle (LSP)

Derived classes are substitutable for base classes:

- All models implement `ModelInterface`
- Can swap baseline/DistilBERT seamlessly
- Polymorphic behavior guaranteed

**Example**:
```python
def predict_sentiment(model: ModelInterface, text: str):
    """Works with any model implementation."""
    return model.predict(text)

# Both work identically
baseline = BaselineModel()
distilbert = DistilBERTModel()
```

### 4. Interface Segregation Principle (ISP)

Clients don't depend on unused interfaces:

- Separate interfaces for loading and cleaning
- Minimal, focused contracts
- No "fat" interfaces

**Example**:
```python
class DataLoaderInterface(ABC):
    """Only loading methods."""
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

class TextCleanerInterface(ABC):
    """Only cleaning methods."""
    @abstractmethod
    def clean(self, text: str) -> str:
        pass
```

### 5. Dependency Inversion Principle (DIP)

Depend on abstractions, not concretions:

- High-level modules depend on interfaces
- Configuration-driven dependency injection
- Loose coupling throughout

**Example**:
```python
class DataProcessor:
    def __init__(
        self,
        loader: DataLoaderInterface,  # Depends on interface
        cleaner: TextCleanerInterface  # Not concrete implementation
    ):
        self.loader = loader
        self.cleaner = cleaner
```

## Design Patterns Used

### 1. Strategy Pattern
Different model strategies (baseline, DistilBERT) implementing same interface.

### 2. Template Method Pattern
BaseModel defines structure, subclasses implement details.

### 3. Dependency Injection
Dependencies injected via constructor, not created internally.

### 4. Singleton Pattern
Configuration and model service as singleton instances.

### 5. Factory Pattern
Model creation based on configuration type.

## Security Implementation

### 1. Authentication
- API Key-based authentication
- Header-based key transmission
- Environment variable configuration

### 2. Authorization
- Role-based access (can be extended)
- Endpoint-level protection

### 3. Input Validation
- Pydantic models for automatic validation
- Length limits and type checking
- SQL injection prevention (no SQL used)
- XSS prevention through sanitization

### 4. Rate Limiting
- IP-based rate limiting
- Configurable limits per endpoint
- Protection against DoS attacks

### 5. Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection
- Strict-Transport-Security
- Content-Security-Policy

### 6. Docker Security
- Non-root user execution
- Minimal base image
- Read-only model volumes
- Health checks

## Data Flow

### Training Flow
```
Raw Data → Load (chunked) → Clean → Balance → Split → Train → Evaluate → Save
                                                           ↓
                                                      MLflow Tracking
```

### Prediction Flow
```
HTTP Request → Validate → Authenticate → Load Model → Preprocess → Predict → Format → Response
```

## Performance Optimizations

1. **Chunked Data Loading**: Handle large files without memory issues
2. **Batch Processing**: Efficient batch predictions
3. **Model Caching**: Model loaded once at startup
4. **Connection Pooling**: Efficient HTTP handling
5. **Async Operations**: Non-blocking I/O where possible

## Scalability Considerations

1. **Horizontal Scaling**: Stateless API, can run multiple instances
2. **Load Balancing**: Docker Compose can be configured with reverse proxy
3. **Caching Layer**: Can add Redis for prediction caching
4. **Message Queue**: Can add Celery for async processing
5. **Database**: Can add PostgreSQL for persistent storage

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock external dependencies
- High coverage (>85%)

### Integration Tests
- API endpoint testing
- End-to-end workflows
- Real component interaction

### Performance Tests
- Load testing capabilities
- Response time benchmarks
- Resource usage monitoring

## Monitoring & Observability

1. **Logging**: Structured logging at all levels
2. **Health Checks**: Endpoint for service monitoring
3. **Metrics**: MLflow for model performance
4. **Tracing**: Request/response logging

## Configuration Management

- Environment variables for configuration
- Pydantic Settings for validation
- Separate dev/prod configurations
- Secret management via .env

## Error Handling Strategy

1. **Validation Errors**: 422 with details
2. **Authentication Errors**: 401 with message
3. **Rate Limit Errors**: 429 with retry info
4. **Server Errors**: 500 with safe message
5. **Model Errors**: 503 when model unavailable

## Future Enhancements

1. **Caching**: Redis for prediction caching
2. **Database**: PostgreSQL for request logging
3. **Message Queue**: Celery for async tasks
4. **A/B Testing**: Gradual model rollout
5. **Model Versioning**: Multiple model versions
6. **Multi-tenancy**: Support for multiple clients
7. **GraphQL**: Alternative API interface
8. **WebSocket**: Real-time predictions
9. **Model Monitoring**: Drift detection
10. **Auto-scaling**: Kubernetes deployment

## Deployment Options

### 1. Docker Compose (Development/Small Scale)
```bash
docker-compose up -d
```

### 2. Kubernetes (Production/Large Scale)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-api
spec:
  replicas: 3
  # ... configuration
```

### 3. Cloud Services
- AWS ECS/EKS
- Google Cloud Run/GKE
- Azure Container Instances/AKS

## Compliance & Standards

- **PEP 8**: Python style guide
- **OpenAPI**: API documentation standard
- **REST**: Architectural style
- **Semantic Versioning**: Version numbering
- **Conventional Commits**: Commit message format

## Documentation

1. **Code Documentation**: Docstrings for all public APIs
2. **API Documentation**: Auto-generated OpenAPI docs
3. **User Documentation**: Comprehensive README
4. **Architecture Documentation**: This file
5. **Examples**: Usage examples for common scenarios
