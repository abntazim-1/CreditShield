# Credit Risk Prediction System

A comprehensive machine learning-based credit risk assessment system built with Flask, featuring a modern web interface for real-time credit risk predictions.

## 🚀 Features

- **Real-time Credit Risk Assessment**: Instant predictions using trained ML models
- **Modern Web Interface**: Beautiful, responsive UI built with HTML5, CSS3, and JavaScript
- **Comprehensive Validation**: Input validation and error handling
- **Risk Level Classification**: LOW, MEDIUM, and HIGH risk categorization
- **Professional Recommendations**: Actionable insights based on risk assessment
- **API Endpoints**: RESTful API for integration with other systems
- **Health Monitoring**: Built-in health checks and model status monitoring

## 🏗️ Architecture

```
Credit_risk_Project/
├── app.py                 # Main Flask application
├── config.py             # Configuration management
├── run.py                # Application runner script
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Web interface template
├── src/                  # ML pipeline components
│   ├── components/       # Data processing components
│   └── pipeline/         # Training pipeline
└── artifacts/            # Trained models and preprocessors
    ├── model.pkl         # Trained ML model
    └── preprocessor.pkl  # Data preprocessor
```

## 📋 Prerequisites

- Python 3.7+
- pip (Python package installer)
- Trained machine learning model files in `artifacts/` directory

## 🛠️ Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <your-repository-url>
   cd Credit_risk_Project
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files exist**:
   - `artifacts/model.pkl` - Your trained machine learning model
   - `artifacts/preprocessor.pkl` - Your data preprocessor

## 🚀 Running the Application

### Option 1: Using the runner script (Recommended)
```bash
python run.py
```

### Option 2: Direct Flask execution
```bash
python app.py
```

### Option 3: Using Flask CLI
```bash
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
```

The application will start at `http://localhost:5000`

## 🌐 Web Interface

### Main Features
- **Personal Information**: Age, income, home ownership, employment length
- **Loan Details**: Intent, grade, amount, interest rate, income percentage
- **Credit History**: Previous defaults, credit history length
- **Real-time Results**: Instant risk assessment with visual feedback

### Risk Levels
- **🟢 LOW RISK**: Loan approval recommended with standard terms
- **🟡 MEDIUM RISK**: Approval with enhanced due diligence
- **🔴 HIGH RISK**: Loan approval not recommended

## 🔌 API Endpoints

### 1. Credit Risk Prediction
- **URL**: `/predict`
- **Method**: `POST`
- **Input**: Form data with credit risk features
- **Output**: JSON with risk assessment and recommendations

### 2. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Output**: Application and model status

### 3. Model Information
- **URL**: `/model-info`
- **Method**: `GET`
- **Output**: Details about the loaded ML model

## 📊 Input Fields

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `person_age` | Integer | Applicant age | 18-100 years |
| `person_income` | Float | Annual income | > 0 |
| `person_home_ownership` | String | Home ownership status | RENT/OWN/MORTGAGE/OTHER |
| `person_emp_length` | Float | Employment length | 0-50 years |
| `loan_intent` | String | Purpose of loan | Various loan purposes |
| `loan_grade` | String | Loan grade | A-G |
| `loan_amnt` | Float | Loan amount | > 0 |
| `loan_int_rate` | Float | Interest rate | 0-100% |
| `loan_percent_income` | Float | Loan as % of income | 0-100% |
| `cb_person_default_on_file` | String | Previous default | Y/N |
| `cb_person_cred_hist_length` | Integer | Credit history length | 0-50 years |

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development`, `production`, or `testing`
- `SECRET_KEY`: Secret key for production (set this in production!)

### Configuration Files
- `config.py`: Centralized configuration management
- Supports different environments (dev, prod, test)

## 📝 Usage Examples

### Making a Prediction via API
```bash
curl -X POST http://localhost:5000/predict \
  -F "person_age=30" \
  -F "person_income=50000" \
  -F "person_home_ownership=RENT" \
  -F "person_emp_length=5" \
  -F "loan_intent=EDUCATION" \
  -F "loan_grade=B" \
  -F "loan_amnt=10000" \
  -F "loan_int_rate=8.5" \
  -F "loan_percent_income=20" \
  -F "cb_person_default_on_file=N" \
  -F "cb_person_cred_hist_length=8"
```

### Health Check
```bash
curl http://localhost:5000/health
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not loaded error**:
   - Ensure `artifacts/model.pkl` and `artifacts/preprocessor.pkl` exist
   - Check file permissions

2. **Port already in use**:
   - Change port in `app.py` or `run.py`
   - Kill existing processes on port 5000

3. **Import errors**:
   - Activate virtual environment
   - Install missing dependencies: `pip install -r requirements.txt`

4. **Template not found**:
   - Ensure `templates/` directory exists
   - Check file paths and permissions

### Debug Mode
Set `FLASK_ENV=development` for detailed error messages and auto-reload.

## 🚀 Production Deployment

### Using Gunicorn (Linux/macOS)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Waitress (Windows)
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Environment Variables for Production
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secure-secret-key
```

## 📈 Monitoring and Logging

- **Application logs**: Check `logs/app.log`
- **Health endpoint**: Monitor `/health` for system status
- **Model status**: Use `/model-info` to verify model loading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Open an issue in the repository

---

**Built with ❤️ using Flask, Machine Learning, and modern web technologies**
