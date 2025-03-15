# Auto Price Prediction: ML Project Structure
This repository contains a structured machine learning project for predicting car prices. The project uses various regression models and provides a unified pipeline for training, evaluation, and prediction.

## Installation
### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup
1. Clone the repository:
```bash
git clone https://github.com/julianbeese/nova_machine_learning.git
cd auto_price_prediction
```
2. Create a virtual environment:
```bash
# With venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Or with conda
conda create -n auto_price_env python=3.8
conda activate auto_price_env
```
3. Install dependencies:
```bash
pip install -r requirements.txt
# Or install as a Python package
pip install -e .
```

## Usage
The project provides a unified command-line interface through the `main.py` file. Here are the main commands:

### Train a Model
To train a single model:
```bash
python main.py --mode train --model elastic_net --input src/data/raw/combined.csv
```
To train all models and select the best one:
```bash
python main.py --mode train --model all --input src/data/raw/combined.csv
```

### Evaluate Models
#### Evaluate a Single Model
To evaluate a specific trained model (the input data must contain a `price` column):
```bash
python main.py --mode evaluate --model elastic_net --input data/raw/auto_prices.csv
```
A detailed evaluation report will be saved in `results/reports/{model_name}_{timestamp}/report.md` with performance metrics and visualizations.

#### Compare All Models
To compare all trained models and generate a comprehensive comparison report:
```bash
python main.py --mode evaluate --model all --input data/raw/auto_prices.csv
```
This will evaluate all available trained models and create a comparison report at `results/reports/model_comparison_{timestamp}/model_comparison_report.md` with performance metrics and visualization charts to easily identify the best-performing model.

### Make Predictions
To make predictions for new data using a trained model:
```bash
python main.py --mode predict --model elastic_net --input data/raw/new_cars.csv
```
Results will be saved in `results/predictions.csv`.

#### Make Predictions using the Streamlit App
```bash
python streamlit run app.py
```

### Command Line Arguments
| Argument | Description | Possible Values | Default |
|----------|-------------|-----------------|---------|
| `--mode` | Operation mode | `train`, `evaluate`, `predict` | `train` |
| `--model` | Model type | `ridge`, `lasso`, `elastic_net`, `random_forest`, `xgboost`, `all`, `best` | `all` |
| `--input` | Path to input file | Path to CSV file | *required* |
| `--output` | Path to save the model | Path to PKL file | `models/{model}_model.pkl` |
| `--log-transform` | Apply logarithmic transformation | Flag (no values) | `False` |

## Models
The project supports several regression models:
1. **Ridge Regression**: Linear regression with L2 regularization
   - Good for datasets with multicollinearity
2. **Lasso Regression**: Linear regression with L1 regularization
   - Performs feature selection (sets unimportant coefficients to zero)
3. **Elastic Net**: Combination of Ridge and Lasso
   - Offers the benefits of both forms of regularization
4. **Random Forest**: Ensemble of decision trees
   - Handles non-linear relationships and feature interactions well
5. **XGBoost**: Gradient Boosting Framework
   - Powerful non-linear model for complex relationships

## Data Format
The input data should be provided as a CSV file with columns for vehicle features. For training and evaluation, a `price` column must be present containing the target value.
Example data format:
```
id,price,brand,model,prod_year,engine_volume,mileage,leather_interior,fuel_type,transmission
1,18000,Toyota,Corolla,2019,1.8,15000,Yes,Gasoline,Automatic
2,12500,Ford,Focus,2017,2.0,35000,No,Diesel,Manual
...
```

## Example Workflow
Here's a typical workflow to use the project:
1. **Data Exploration**: Use the notebooks to analyze your data
2. **Model Training**: Train all models to find the best one
   ```bash
   python main.py --mode train --model all --input src/data/raw/combined.csv
   ```
3. **Model Comparison**: Compare all models to find the best performer
   ```bash
   python main.py --mode evaluate --model all --input src/data/raw/combined.csv
   ```
4. **Detailed Evaluation**: Get detailed insights on your chosen model
   ```bash
   python main.py --mode evaluate --model elastic_net --input src/data/raw/combined.csv
   ```
5. **Predictions**: Use the best model for predictions on new data
   ```bash
   python main.py --mode predict --model best --input src/data/raw/new_cars.csv
   ```
6. **Predictions**: Use the best model for predictions on new data in the streamlit app
   ```bash
   python streamlit run app.py
   ```
