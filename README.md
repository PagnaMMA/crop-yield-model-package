# Fertilizer Prediction Model

A machine learning model package for predicting the yield for a particular crop base on certains conditions.

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/PagnaMMA/crop-yield-model-package.git
```

## Usage
```python
from cropyield_model import YieldPredictor

# Initialize the predictor
predictor = YieldPredictor()

# Make predictions
features = [30.0, 200.0, 7.0, 0.70, 75.0, 85.0, 70.0, 'Alkaline Soil', 'wheat']  # Example features
prediction = predictor.predict_yield(features)
print(f"Predicted fertilizer: {prediction}")
```

## Model Details

- Model type: [Linear Regression]
- Features: [Soil_type, crop, rainfall_mm,temperature_celsius, fertilizer_used, irrigation_used, weather_condition, days_to_harvest]
- Target: Yield_tons_hectare

## Version History

- 0.1.0: Initial release
```

## Step 9: Create `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db