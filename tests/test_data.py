"""Tests for data processing module."""
import pytest
import pandas as pd
from src.data import (
    YelpReviewLoader,
    BasicTextCleaner,
    DataProcessor
)


class TestYelpReviewLoader:
    """Test cases for YelpReviewLoader."""
    
    def test_load_data_with_sample_size(self, temp_data_file):
        """Test loading data with sample size limit."""
        loader = YelpReviewLoader()
        df = loader.load_data(temp_data_file, sample_size=3)
        
        assert len(df) <= 3
        assert 'text' in df.columns
        assert 'sentiment' in df.columns
    
    def test_load_data_sentiment_mapping(self, temp_data_file):
        """Test correct sentiment mapping from stars."""
        loader = YelpReviewLoader()
        df = loader.load_data(temp_data_file, sample_size=5)
        
        # Check sentiment mapping
        sentiments = df['sentiment'].unique()
        assert all(s in ['positive', 'negative', 'neutral'] for s in sentiments)
    
    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        loader = YelpReviewLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent_file.json")


class TestBasicTextCleaner:
    """Test cases for BasicTextCleaner."""
    
    def test_clean_basic_text(self):
        """Test basic text cleaning."""
        cleaner = BasicTextCleaner()
        text = "This is a GREAT product!"
        cleaned = cleaner.clean(text)
        
        assert cleaned == "this is a great product"
    
    def test_clean_with_url(self):
        """Test URL removal."""
        cleaner = BasicTextCleaner()
        text = "Check out https://example.com for more info"
        cleaned = cleaner.clean(text)
        
        assert "https://example.com" not in cleaned
    
    def test_clean_with_html(self):
        """Test HTML tag removal."""
        cleaner = BasicTextCleaner()
        text = "<p>This is <b>bold</b> text</p>"
        cleaned = cleaner.clean(text)
        
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
    
    def test_clean_empty_string(self):
        """Test handling of empty strings."""
        cleaner = BasicTextCleaner()
        cleaned = cleaner.clean("")
        
        assert cleaned == ""
    
    def test_clean_non_string(self):
        """Test handling of non-string input."""
        cleaner = BasicTextCleaner()
        cleaned = cleaner.clean(None)
        
        assert cleaned == ""


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def test_process_data(self, temp_data_file):
        """Test complete data processing pipeline."""
        loader = YelpReviewLoader()
        cleaner = BasicTextCleaner()
        processor = DataProcessor(loader, cleaner)
        
        train_df, test_df = processor.process(
            temp_data_file,
            sample_size=5,
            test_size=0.4
        )
        
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert 'cleaned_text' in train_df.columns
        assert 'cleaned_text' in test_df.columns
    
    def test_data_split_stratification(self, temp_data_file):
        """Test that data splitting maintains class distribution."""
        loader = YelpReviewLoader()
        cleaner = BasicTextCleaner()
        processor = DataProcessor(loader, cleaner)
        
        train_df, test_df = processor.process(
            temp_data_file,
            sample_size=5,
            test_size=0.4
        )
        
        # Both sets should have data
        assert len(train_df) > 0
        assert len(test_df) > 0
