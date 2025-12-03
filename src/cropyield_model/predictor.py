import os

from pathlib import Path
import numpy as np
import joblib

class YieldPredictor:
    """
    A class to load,make predictions with the trained crop_yield prediction model.
    And finally display the results
    """
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        # Models are in the same directory as this file
        self.model_path = Path(__file__).parent / 'crop_model_Linear_Regression.pkl'
        self.encoders_path = Path(__file__).parent / 'label_encoders.pkl'
        self.scaler_path = Path(__file__).parent / 'scaler_crop.pkl'
    
    def load_model(self):
        """Load trained model, encoders, and scaler"""
        if self.model_path.exists() and self.encoders_path.exists() and self.scaler_path.exists():
            self.model = joblib.load(self.model_path)
            encoders = joblib.load(self.encoders_path)
            self.label_encoders = encoders['label_encoders']
            self.scaler = joblib.load(self.scaler_path)
            print("Model loaded successfully")
            return True
        else:
            print(f"Model files not found:")
            print(f"  Model: {self.model_path.exists()}")
            print(f"  Encoders: {self.encoders_path.exists()}")
            print(f"  Scaler: {self.scaler_path.exists()}")
        return False
    
    def predict_yield(self, input_data):
        """
        Predict crop yield for given conditions
        Parameters:
        -----------
        input_data : dict
            Input features including:
            - region (str)
            - soil_type (str)
            - crop (str)
            - rainfall_mm (float)
            - temperature_celsius (float)
            - fertilizer_used (bool)
            - irrigation_used (bool)
            - weather_condition (str)
            - days_to_harvest (int)
        Returns:
        --------
        Float : predicted_yield amount in tons/hectare
        """

        # Extract input features
        region = input_data.get('region')
        soil_type = input_data.get('soil_type')
        crop = input_data.get('crop')
        rainfall_mm = input_data.get('rainfall_mm')
        temperature_celsius = input_data.get('temperature_celsius')
        fertilizer_used = input_data.get('fertilizer_used')
        irrigation_used = input_data.get('irrigation_used')
        weather_condition = input_data.get('weather_condition')
        days_to_harvest = input_data.get('days_to_harvest')

        # Load model if not loaded
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained. Please train the model first.")

        # Encode categorical variables
        try:
            region_encoded = self.label_encoders['Region'].transform([region])[0]
        except ValueError:
            region_encoded = 0

        try:
            soil_type_encoded = self.label_encoders['Soil_Type'].transform([soil_type])[0]
        except ValueError:
            soil_type_encoded = 0

        try:
            crop_encoded = self.label_encoders['Crop'].transform([crop])[0]
        except ValueError:
            crop_encoded = 0

        try:
            weather_encoded = self.label_encoders['Weather_Condition'].transform([weather_condition])[0]
        except ValueError:
            weather_encoded = 0

        # Convert boolean to integer
        fertilizer_used_int = 1 if fertilizer_used else 0
        irrigation_used_int = 1 if irrigation_used else 0

        # Prepare features
        features = np.array([[region_encoded, soil_type_encoded, crop_encoded,
                            rainfall_mm, temperature_celsius, fertilizer_used_int,
                            irrigation_used_int, weather_encoded, days_to_harvest]])

        # Scale features (Linear Regression was trained with scaled data)
        features_scaled = self.scaler.transform(features)

        # Get prediction
        predicted_yield = self.model.predict(features_scaled)[0]

        return float(round(predicted_yield, 2))
        

    def display_result(self, input_data):
        """
        Display the input data and prediction results.
        Parameters:
        -----------
        input_data : dict
            Input features.
        predicted_yield : float
            Predicted yield amount in tons/hectare.
        """
        
        print("\n" + "="*70)
        print("Predictions Results")
        print("="*70)
        region = input_data.get('region')
        soil_type = input_data.get('soil_type')
        crop = input_data.get('crop')
        rainfall_mm = input_data.get('rainfall_mm')
        temperature_celsius = input_data.get('temperature_celsius')
        fertilizer_used = input_data.get('fertilizer_used')
        irrigation_used = input_data.get('irrigation_used')
        weather_condition = input_data.get('weather_condition')
        days_to_harvest = input_data.get('days_to_harvest')
        print(f"\n ==> Input Data:")
        print(f" - Temperature: {temperature_celsius} °C")
        print(f" - Rainfall: {rainfall_mm} mm")
        print(f" - Soil Type: {soil_type}")
        print(f" - Crop: {crop}")
        print(f" - Region: {region}")
        print(f" - Fertilizer Used: {'Yes' if fertilizer_used else 'No'}")
        print(f" - Irrigation Used: {'Yes' if irrigation_used else 'No'}")
        print(f" - Weather Condition: {weather_condition}")
        print(f" - Days to Harvest: {days_to_harvest} days")
        print("\n" + "="*70)
        print(f"\n ==> Predictions:")
        try:
            predicted_yield = self.predict_yield(input_data)
            print(f"\n Predicted Yield: {predicted_yield} tons/hectare")
        except Exception as e:
            print(f" Error: {e}")

