"""
Data Loading and Preprocessing Module
Handles loading and basic preprocessing of MovieLens datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class DataLoader:
    """Load and preprocess MovieLens data"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Path to directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        self.movies_df = None
        self.tags_df = None
        self.genome_scores_df = None
        self.genome_tags_df = None
        
    def load_all_data(self) -> Dict:
        """
        Load all dataset files
        
        Returns:
            Dictionary containing all loaded dataframes
        """
        print("Loading MovieLens datasets...")
        
        # Load main files
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.test_df = pd.read_csv(self.data_dir / "test.csv")
        self.movies_df = pd.read_csv(self.data_dir / "movies.csv")
        self.tags_df = pd.read_csv(self.data_dir / "tags.csv")
        self.genome_scores_df = pd.read_csv(self.data_dir / "genome_scores.csv")
        self.genome_tags_df = pd.read_csv(self.data_dir / "genome_tags.csv")
        
        print(f"✓ Training data: {len(self.train_df)} ratings")
        print(f"✓ Test data: {len(self.test_df)} pairs")
        print(f"✓ Movies: {len(self.movies_df)} movies")
        print(f"✓ Tags: {len(self.tags_df)} tag entries")
        print(f"✓ Genome scores: {len(self.genome_scores_df)} entries")
        
        return {
            'train': self.train_df,
            'test': self.test_df,
            'movies': self.movies_df,
            'tags': self.tags_df,
            'genome_scores': self.genome_scores_df,
            'genome_tags': self.genome_tags_df
        }
    
    def get_data_info(self) -> None:
        """Print information about loaded datasets"""
        if self.train_df is not None:
            print("\n=== TRAINING DATA ===")
            print(f"Shape: {self.train_df.shape}")
            print(f"Unique users: {self.train_df['userId'].nunique()}")
            print(f"Unique movies: {self.train_df['movieId'].nunique()}")
            print(f"Rating range: {self.train_df['rating'].min()} - {self.train_df['rating'].max()}")
            print(f"Missing values:\n{self.train_df.isnull().sum()}")
            
        if self.test_df is not None:
            print("\n=== TEST DATA ===")
            print(f"Shape: {self.test_df.shape}")
            print(f"Unique users: {self.test_df['userId'].nunique()}")
            print(f"Unique movies: {self.test_df['movieId'].nunique()}")
    
    def get_sparsity(self) -> float:
        """
        Calculate sparsity of rating matrix
        
        Returns:
            Sparsity percentage (0-100)
        """
        if self.train_df is None:
            return None
            
        n_users = self.train_df['userId'].nunique()
        n_movies = self.train_df['movieId'].nunique()
        n_ratings = len(self.train_df)
        
        total_possible = n_users * n_movies
        sparsity = (1 - n_ratings / total_possible) * 100
        
        print(f"Matrix sparsity: {sparsity:.2f}%")
        print(f"Total possible ratings: {total_possible:,}")
        print(f"Actual ratings: {n_ratings:,}")
        
        return sparsity


def load_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load train and test data
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Tuple of (train_df, test_df)
    """
    loader = DataLoader(data_dir)
    data = loader.load_all_data()
    return data['train'], data['test']


if __name__ == "__main__":
    # Example usage
    loader = DataLoader("data")
    data = loader.load_all_data()
    loader.get_data_info()
    loader.get_sparsity()
