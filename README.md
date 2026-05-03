# MovieLens Movie Recommendation System 🎬

A machine learning project for predicting movie ratings using the MovieLens dataset. This is a collaborative filtering and content-based recommendation system built for the Kaggle MovieLens competition.

## 📊 Project Overview

This project implements multiple recommendation approaches:
- **Baseline Models**: Average rating, user/movie bias
- **Collaborative Filtering**: User-based and item-based approaches
- **Matrix Factorization**: SVD-based dimensionality reduction
- **Hybrid Approaches**: Combining multiple techniques

## 📁 Dataset

The dataset contains millions of 5-star ratings from the [MovieLens recommendation service](https://grouplens.org/datasets/movielens/):

### Files Included:
- `train.csv` - Training ratings (userId, movieId, rating, timestamp)
- `test.csv` - Test set (userId, movieId pairs without ratings)
- `movies.csv` - Movie metadata (title, genres)
- `tags.csv` - User-generated tags
- `genome_scores.csv` - Movie-tag relevance scores
- `genome_tags.csv` - Tag descriptions
- `links.csv` - IMDB/TMDB IDs
- `imdb_data.csv` - IMDB metadata

### Key Statistics:
- **Data Size**: ~878 MB
- **Format**: 5-star ratings (0.5 - 5.0 with 0.5 increments)
- **Timestamps**: Unix format (seconds since 1970-01-01)

## 🚀 Quick Start

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/Yuzzitech/MovieLens-Recommendation-2026.git
cd MovieLens-Recommendation-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
# 1. Download data from Kaggle competition
# 2. Place CSV files in data/ directory

# 3. Run exploratory analysis
jupyter notebook notebooks/01_EDA.ipynb

# 4. Preprocess data
jupyter notebook notebooks/02_preprocessing.ipynb

# 5. Train models
jupyter notebook notebooks/03_modeling.ipynb

# 6. Make predictions
python src/predict.py
```

## 📁 Project Structure

```
MovieLens-Recommendation-2026/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── data/                        # Data files (not tracked in git)
│   └── .gitkeep
├── notebooks/                   # Jupyter notebooks
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   # Data cleaning & feature engineering
│   └── 03_modeling.ipynb        # Model training & evaluation
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_loader.py          # Load and preprocess data
│   ├── models.py               # ML models and algorithms
│   ├── evaluation.py           # Metrics and evaluation
│   ├── utils.py                # Helper functions
│   └── predict.py              # Make predictions
├── submissions/                 # Prediction submissions
│   └── .gitkeep
└── config.yaml                 # Configuration file
```

## 🔧 Technology Stack

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - ML models
- **scipy** - Scientific computing
- **jupyter** - Interactive notebooks
- **matplotlib/seaborn** - Visualization

## 📈 Methodology

### Phase 1: Exploratory Data Analysis (EDA)
- Load and inspect all datasets
- Analyze rating distributions
- Explore user/movie patterns
- Check data quality and missing values
- Visualize sparsity patterns

### Phase 2: Data Preprocessing
- Handle missing values
- Feature engineering
- Create user/movie embeddings
- Merge additional features
- Train/validation split

### Phase 3: Model Development
1. **Baseline Models**
   - Global mean rating
   - User bias + Movie bias
   
2. **Collaborative Filtering**
   - User-based CF
   - Item-based CF
   
3. **Matrix Factorization**
   - SVD (Singular Value Decomposition)
   - NMF (Non-negative Matrix Factorization)
   
4. **Ensemble Methods**
   - Combine multiple models
   - Weighted averaging

### Phase 4: Evaluation & Submission
- Cross-validation
- RMSE metric optimization
- Hyperparameter tuning
- Final predictions on test set

## 📊 Expected Results

- **Baseline RMSE**: ~1.0-1.2
- **Target RMSE**: < 0.85 (competitive)

## 🎯 Competition Goals

- Predict movie ratings for user-movie pairs
- Minimize RMSE (Root Mean Squared Error)
- Rank well on Kaggle leaderboard
- Learn collaborative filtering techniques

## 📝 Notes for Beginners

This project is ideal for learning:
- ✅ Data exploration with pandas
- ✅ Data cleaning and preprocessing
- ✅ Feature engineering
- ✅ Collaborative filtering algorithms
- ✅ Model evaluation and validation
- ✅ Kaggle competition workflow

## 🤝 Contributing

Feel free to:
- Open issues for bugs or improvements
- Submit pull requests with enhancements
- Share better model architectures
- Suggest optimization techniques

## 📚 Resources

- [MovieLens Dataset Documentation](https://grouplens.org/datasets/movielens/)
- [Collaborative Filtering Tutorial](https://developers.google.com/machine-learning/recommendation)
- [Kaggle Competitions Guide](https://www.kaggle.com/docs/competitions)

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Happy coding!** 🚀 Feel free to reach out with questions or improvements.
