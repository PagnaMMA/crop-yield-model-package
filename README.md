# Crop Yield Prediction Model

A machine learning model package for predicting the yield for a particular crop based on certain conditions.

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
# Example 1: Rice in South region
features_data_1 = {
    'region': 'South',
    'soil_type': 'Clay',
    'crop': 'Rice',
    'rainfall_mm': 992.7,
    'temperature_celsius': 18.0,
    'fertilizer_used': True,
    'irrigation_used': True,
    'weather_condition': 'Rainy',
    'days_to_harvest': 140
}
# Example 2: Cotton in West region
features_data_2 = {
    'region': 'West',
    'soil_type': 'Sandy',
    'crop': 'Cotton',
    'rainfall_mm': 897.1,
    'temperature_celsius': 27.7,
    'fertilizer_used': False,
    'irrigation_used': True,
    'weather_condition': 'Cloudy',
    'days_to_harvest': 122
}

# Example 3: Maize in East region
features_data_3 = {
    'region': 'East',
    'soil_type': 'Silt',
    'crop': 'Maize',
    'rainfall_mm': 450.0,
    'temperature_celsius': 30.0,
    'fertilizer_used': True,
    'irrigation_used': False,
    'weather_condition': 'Sunny',
    'days_to_harvest': 95
}
prediction = predictor.predict_yield(features_data_2)
print(f"Predicted yield: {prediction} tons/hectare")
```

## Model Details

- **Model type**: Linear Regression
- **Features**: Region, Soil Type, Crop, Rainfall (mm), Temperature (°C), Fertilizer Used, Irrigation Used, Weather Condition, Days to Harvest
- **Target**: Yield (tons/hectare)
- **Training Data**: 1,000,000 samples
- **Model Performance**: R² score, RMSE metrics available

## Valid Feature Values

- **Region**: 'North', 'South', 'East', 'West'
- **Soil Type**: 'Sandy', 'Clay', 'Loam', 'Silt'
- **Crop**: 'Wheat', 'Rice', 'Maize', 'Barley', 'Cotton', 'Soybean'
- **Rainfall**: 100-1000 mm
- **Temperature**: 15-40°C
- **Fertilizer Used**: True/False
- **Irrigation Used**: True/False
- **Weather Condition**: 'Sunny', 'Rainy', 'Cloudy'
- **Days to Harvest**: 60-149 days

## Version History

- 0.1.0: Initial release

## License

MIT License