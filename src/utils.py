"""
Utility Functions
Helper functions for data processing and analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path


def ensure_directory(path: str) -> Path:
    """Create directory if it doesn't exist"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_user_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each user
    
    Args:
        train_df: Training dataframe with userId, rating columns
        
    Returns:
        DataFrame with user statistics
    """
    stats = train_df.groupby('userId')['rating'].agg([
        ('count', 'size'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    return stats.sort_values('count', ascending=False)


def get_movie_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each movie
    
    Args:
        train_df: Training dataframe with movieId, rating columns
        
    Returns:
        DataFrame with movie statistics
    """
    stats = train_df.groupby('movieId')['rating'].agg([
        ('count', 'size'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    return stats.sort_values('count', ascending=False)


def create_rating_matrix(train_df: pd.DataFrame, 
                        user_col: str = 'userId',
                        item_col: str = 'movieId',
                        rating_col: str = 'rating',
                        sparse: bool = False):
    """
    Create user-item rating matrix
    
    Args:
        train_df: Training dataframe
        user_col: Name of user column
        item_col: Name of item column
        rating_col: Name of rating column
        sparse: Return sparse matrix if True
        
    Returns:
        Rating matrix (dense or sparse)
    """
    matrix = train_df.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col,
        fill_value=0
    )
    
    if sparse:
        from scipy.sparse import csr_matrix
        return csr_matrix(matrix)
    
    return matrix


def clip_predictions(predictions: np.ndarray,
                    min_rating: float = 0.5,
                    max_rating: float = 5.0) -> np.ndarray:
    """
    Clip predictions to valid rating range
    
    Args:
        predictions: Array of predicted ratings
        min_rating: Minimum valid rating
        max_rating: Maximum valid rating
        
    Returns:
        Clipped predictions
    """
    return np.clip(predictions, min_rating, max_rating)


def prepare_submission(predictions: np.ndarray,
                      test_df: pd.DataFrame,
                      output_path: str = "submissions/submission.csv") -> pd.DataFrame:
    """
    Prepare predictions for Kaggle submission
    
    Args:
        predictions: Array of predicted ratings
        test_df: Test dataframe with userId, movieId
        output_path: Path to save submission file
        
    Returns:
        Submission dataframe
    """
    submission = test_df.copy()
    submission['rating'] = clip_predictions(predictions)
    
    # Ensure output directory exists
    ensure_directory(Path(output_path).parent)
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"✓ Submission saved to {output_path}")
    
    return submission


def save_model_results(results: dict, output_path: str = "submissions/results.txt"):
    """
    Save model evaluation results
    
    Args:
        results: Dictionary of results
        output_path: Path to save results
    """
    ensure_directory(Path(output_path).parent)
    
    with open(output_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
