#!/usr/bin/env python3
"""
Credit Risk Prediction System - Flask Application Runner
"""

import os
import sys
from app import app, load_model_and_preprocessor
from config import get_config

def main():
    """Main function to run the Flask application"""
    
    # Set environment
    os.environ['FLASK_ENV'] = 'development'
    
    # Load configuration
    config = get_config()
    app.config.from_object(config)
    
    # Load model and preprocessor
    print("🚀 Starting Credit Risk Prediction System...")
    print("📊 Loading machine learning model and preprocessor...")
    
    load_model_and_preprocessor()
    
    # Check if model files exist
    if not os.path.exists(config.MODEL_PATH):
        print(f"⚠️  Warning: Model file not found at {config.MODEL_PATH}")
        print("   Please ensure you have trained the model first.")
    
    if not os.path.exists(config.PREPROCESSOR_PATH):
        print(f"⚠️  Warning: Preprocessor file not found at {config.PREPROCESSOR_PATH}")
        print("   Please ensure you have trained the model first.")
    
    print("✅ Application ready!")
    print(f"🌐 Server will start at: http://localhost:5000")
    print("📱 Open your browser and navigate to the URL above")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run the Flask app
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=config.DEBUG
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
