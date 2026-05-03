"""
Evaluation Metrics Module
Functions for evaluating recommendation model performance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Union


def rmse(y_true: Union[np.ndarray, pd.Series], 
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: Actual ratings
        y_pred: Predicted ratings
        
    Returns:
        RMSE value
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def mae(y_true: Union[np.ndarray, pd.Series], 
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        y_true: Actual ratings
        y_pred: Predicted ratings
        
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def clipped_rmse(y_true: Union[np.ndarray, pd.Series], 
                 y_pred: Union[np.ndarray, pd.Series],
                 min_rating: float = 0.5,
                 max_rating: float = 5.0) -> float:
    """
    Calculate RMSE with clipped predictions (within rating range)
    
    Args:
        y_true: Actual ratings
        y_pred: Predicted ratings
        min_rating: Minimum valid rating
        max_rating: Maximum valid rating
        
    Returns:
        Clipped RMSE value
    """
    y_pred_clipped = np.clip(y_pred, min_rating, max_rating)
    return rmse(y_true, y_pred_clipped)


def evaluate_model(y_true: Union[np.ndarray, pd.Series], 
                   y_pred: Union[np.ndarray, pd.Series],
                   verbose: bool = True) -> dict:
    """
    Comprehensive model evaluation
    
    Args:
        y_true: Actual ratings
        y_pred: Predicted ratings
        verbose: Print results if True
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'clipped_rmse': clipped_rmse(y_true, y_pred),
        'mean_true': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_true': np.std(y_true),
        'std_pred': np.std(y_pred)
    }
    
    if verbose:
        print("\n=== Model Evaluation ===")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Clipped RMSE: {metrics['clipped_rmse']:.4f}")
        print(f"\nTrue ratings - Mean: {metrics['mean_true']:.2f}, Std: {metrics['std_true']:.2f}")
        print(f"Pred ratings - Mean: {metrics['mean_pred']:.2f}, Std: {metrics['std_pred']:.2f}")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    y_true = np.array([3.0, 4.0, 2.5, 4.5, 3.5])
    y_pred = np.array([3.2, 3.8, 2.7, 4.3, 3.6])
    
    metrics = evaluate_model(y_true, y_pred)
