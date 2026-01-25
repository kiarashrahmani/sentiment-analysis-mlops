"""
Training script for sentiment analysis models.
Supports both baseline and DistilBERT models.
"""
import argparse
import logging
from pathlib import Path
import json

from config import settings
from src.data import YelpReviewLoader, BasicTextCleaner, DataProcessor
from src.models import BaselineModel, DistilBERTModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    model_type: str = "baseline",
    data_path: str = None,
    sample_size: int = None,
    output_dir: str = "models"
):
    """
    Train a sentiment analysis model.
    
    Args:
        model_type: Type of model ('baseline' or 'distilbert')
        data_path: Path to dataset
        sample_size: Number of samples to use
        output_dir: Directory to save trained model
    """
    logger.info(f"Starting training for {model_type} model")
    
    # Load and process data
    logger.info("Loading and processing data")
    loader = YelpReviewLoader()
    cleaner = BasicTextCleaner()
    processor = DataProcessor(loader, cleaner)
    
    train_df, test_df = processor.process(
        file_path=data_path or settings.data_path,
        sample_size=sample_size or settings.sample_size,
        test_size=settings.test_size,
        random_state=settings.random_state
    )
    
    # Save processed data
    processor.save_processed_data(train_df, test_df)
    
    # Create and train model
    logger.info(f"Creating {model_type} model")
    
    if model_type == "baseline":
        model = BaselineModel(random_state=settings.random_state)
    elif model_type == "distilbert":
        model = DistilBERTModel(
            max_length=settings.max_length,
            batch_size=settings.batch_size,
            random_state=settings.random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.train(train_df['cleaned_text'], train_df['sentiment'])
    
    # Evaluate
    logger.info("Evaluating model")
    metrics = model.evaluate(test_df['cleaned_text'], test_df['sentiment'])
    
    # Print results
    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_type}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
    logger.info(f"{'='*50}\n")
    
    # Save model
    output_path = Path(output_dir) / f"{model_type}_model"
    if model_type == "baseline":
        output_path = output_path.with_suffix('.pkl')
    
    model.save(str(output_path))
    logger.info(f"Model saved to {output_path}")
    
    # Save metrics to file
    metrics_file = Path(output_dir) / f"{model_type}_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump({
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'confusion_matrix': metrics['confusion_matrix']
        }, f, indent=2)
    
    logger.info("Training complete!")
    
    return model, metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train sentiment analysis models")
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["baseline", "distilbert", "both"],
        default="baseline",
        help="Type of model to train"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset (default: from config)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to use (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save models"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train models
    if args.model_type == "both":
        logger.info("Training both models")
        
        # Train baseline first
        logger.info("\n" + "="*60)
        logger.info("Training Baseline Model")
        logger.info("="*60)
        train_model("baseline", args.data_path, args.sample_size, args.output_dir)
        
        # Train DistilBERT
        logger.info("\n" + "="*60)
        logger.info("Training DistilBERT Model")
        logger.info("="*60)
        train_model("distilbert", args.data_path, args.sample_size, args.output_dir)
    else:
        train_model(args.model_type, args.data_path, args.sample_size, args.output_dir)


if __name__ == "__main__":
    main()
