import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_model_and_preprocessor():
    """Load the trained model and preprocessor from artifacts"""
    global model, preprocessor
    
    try:
        # Load the trained model
        model_path = os.path.join('artifacts', 'model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
            
        # Load the preprocessor
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded successfully")
        else:
            logger.error(f"Preprocessor file not found at {preprocessor_path}")
            
    except Exception as e:
        logger.error(f"Error loading model/preprocessor: {str(e)}")

def validate_input_data(data):
    """Validate input data for credit risk prediction"""
    required_fields = [
        'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
        'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_default_on_file', 'cb_person_cred_hist_length'
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate numeric fields
    try:
        age = int(data['person_age'])
        if age < 18 or age > 100:
            return False, "Person age must be between 18 and 100"
            
        income = float(data['person_income'])
        if income < 0:
            return False, "Person income cannot be negative"
            
        emp_length = float(data['person_emp_length'])
        if emp_length < 0 or emp_length > 50:
            return False, "Employment length must be between 0 and 50 years"
            
        loan_amnt = float(data['loan_amnt'])
        if loan_amnt <= 0:
            return False, "Loan amount must be positive"
            
        loan_int_rate = float(data['loan_int_rate'])
        if loan_int_rate < 0 or loan_int_rate > 100:
            return False, "Loan interest rate must be between 0 and 100"
            
        loan_percent_income = float(data['loan_percent_income'])
        if loan_percent_income < 0 or loan_percent_income > 100:
            return False, "Loan percent income must be between 0 and 100"
            
        cred_hist_length = int(data['cb_person_cred_hist_length'])
        if cred_hist_length < 0 or cred_hist_length > 50:
            return False, "Credit history length must be between 0 and 50 years"
            
    except ValueError:
        return False, "Invalid numeric values in input data"
    
    return True, "Data validation successful"

def prepare_features(data):
    """Prepare features for model prediction"""
    # Create feature dictionary with proper data types
    features = {
        'person_age': int(data['person_age']),
        'person_income': float(data['person_income']),
        'person_home_ownership': data['person_home_ownership'],
        'person_emp_length': float(data['person_emp_length']),
        'loan_intent': data['loan_intent'],
        'loan_grade': data['loan_grade'],
        'loan_amnt': float(data['loan_amnt']),
        'loan_int_rate': float(data['loan_int_rate']),
        'loan_percent_income': float(data['loan_percent_income']),
        'cb_person_default_on_file': data['cb_person_default_on_file'],
        'cb_person_cred_hist_length': int(data['cb_person_cred_hist_length'])
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    return df

def get_risk_level_and_recommendation(prediction_proba):
    """Get risk level and recommendation based on prediction probability"""
    if prediction_proba < 0.3:
        risk_level = "LOW RISK"
        risk_color = "#28a745"
        recommendation = "Loan approval recommended with standard terms."
        confidence = "High confidence in approval"
    elif prediction_proba < 0.7:
        risk_level = "MEDIUM RISK"
        risk_color = "#ffc107"
        recommendation = "Loan approval with enhanced due diligence and adjusted terms."
        confidence = "Moderate confidence - requires review"
    else:
        risk_level = "HIGH RISK"
        risk_color = "#dc3545"
        recommendation = "Loan approval not recommended. Consider alternative options."
        confidence = "High confidence in rejection"
    
    return {
        'risk_level': risk_level,
        'risk_color': risk_color,
        'recommendation': recommendation,
        'confidence': confidence,
        'risk_score': f"{prediction_proba * 100:.1f}%"
    }

@app.route('/')
def index():
    """Main page for credit risk prediction"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle credit risk prediction"""
    try:
        if model is None or preprocessor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure model files are available.'
            }), 500
        
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate input data
        is_valid, validation_message = validate_input_data(form_data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': validation_message
            }), 400
        
        # Prepare features
        features_df = prepare_features(form_data)
        
        # Preprocess features
        try:
            processed_features = preprocessor.transform(features_df)
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Error processing input data. Please check your inputs.'
            }), 500
        
        # Make prediction
        try:
            prediction_proba = model.predict_proba(processed_features)[0][1]  # Probability of default
            prediction = model.predict(processed_features)[0]
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Error making prediction. Please try again.'
            }), 500
        
        # Get risk assessment
        risk_assessment = get_risk_level_and_recommendation(prediction_proba)
        
        # Prepare response
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_proba': float(prediction_proba),
            'risk_assessment': risk_assessment,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Prediction successful: {risk_assessment['risk_level']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    preprocessor_status = "loaded" if preprocessor is not None else "not loaded"
    
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'preprocessor': preprocessor_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 404
    
    try:
        model_info = {
            'model_type': type(model).__name__,
            'features': getattr(model, 'feature_names_', 'Unknown'),
            'classes': getattr(model, 'classes_', 'Unknown').tolist() if hasattr(model, 'classes_') else 'Unknown'
        }
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model and preprocessor when starting the app
    print("Loading model and preprocessor...")
    load_model_and_preprocessor()
    
    if model is None or preprocessor is None:
        print("Warning: Model or preprocessor not loaded. Some features may not work.")
        print("Please ensure the following files exist:")
        print("- artifacts/model.pkl")
        print("- artifacts/preprocessor.pkl")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True  # Set to False in production
    )
