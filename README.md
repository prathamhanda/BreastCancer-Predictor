# Breast Cancer Predictor || [Live Preview](https://brcancer.streamlit.app/)

![Breast Cancer Predictor Platform](https://res.cloudinary.com/dglcgpley/image/upload/v1751453007/banner_xplmkk.png)
-- 
A machine learning web application that predicts whether a breast mass is benign or malignant based on measurements from cytology lab results.

## Features
- Interactive web interface with adjustable measurements
- Real-time predictions
- Visual representation of measurements using radar charts
- Professional medical-grade predictions

## Installation

### Prerequisites
1. Python 3.8 or higher
2. Git

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/prathamhanda/BreastCancer-Predictor.git
cd BreastCancer-Predictor
```

2. Create and activate a virtual environment:

For Windows:
```bash
pip install numpy pandas scikit-learn streamlit plotly altair
```

```bash
python -m venv .venv
```

```bash
source .venv/Scripts/activate
```

```bash
python -m pip install streamlit
```

```bash
pip install plotly scikit-learn
```

```bash
python -m streamlit run app/main.py
```

For Linux/Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
# Make sure you're in the project root directory
python -m streamlit run app/main.py
```

5. Open your web browser and go to:
```
http://localhost:8501
```

## Project Structure
- `app/main.py`: Main application file
- `data/data.csv`: Dataset for training
- `model/`: Contains trained model files
- `assets/`: CSS and other static files
- `confusion_matrices/`: Contains model performance visualizations

## Usage
1. Launch the application using the instructions above
2. Use the sliders in the sidebar to adjust measurements
3. View the radar chart visualization
4. Check the prediction results

## Common Issues and Solutions

### 1. Permission Errors When Creating Virtual Environment
If you see an error like:
```
Error: [Errno 13] Permission denied: '...\.venv\Scripts\python.exe'
```

Solutions:
- Close VS Code and any terminals using the `.venv` directory
- Delete the existing `.venv` directory: `rm -rf .venv`
- Run VS Code as administrator
- Create the virtual environment in your user directory instead:
```bash
cd %USERPROFILE%
python -m venv breast-cancer-env
breast-cancer-env\Scripts\activate
cd path/to/Breast-Cancer-Predictor
```

### 2. Streamlit Command Not Found
If you see:
```
bash: streamlit: command not found
```

Solutions:
- Always use `python -m streamlit run app/main.py` instead of `streamlit run app/main.py`
- Make sure you're in the project root directory, not in the app directory
- Verify streamlit is installed: `pip list | grep streamlit`
- Reinstall if needed: `pip install streamlit`

### 3. scikit-learn Version Warnings
If you see warnings like:
```
InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version 1.2.2 when using version 1.6.1
```

Solutions:
- These warnings are expected and won't affect functionality
- If you want to eliminate warnings, install the exact version:
```bash
pip install scikit-learn==1.2.2
```

### 4. Feature Names Warning
If you see:
```
UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
```

Solution:
- This is an expected warning and won't affect the model's predictions
- The warning appears because of scikit-learn version differences

### 5. General Troubleshooting Steps

1. Verify Virtual Environment:
```bash
# Should show (.venv) in prompt
echo %VIRTUAL_ENV%
```

2. Check Installed Packages:
```bash
pip list
```

3. Verify Working Directory:
```bash
# Should be in project root
pwd
```

4. Clean Installation:
```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

5. Port Already in Use:
If port 8501 is in use, Streamlit will automatically try the next available port (8502, etc.)

## Model Performance
The application includes three machine learning models for breast cancer prediction:
- Random Forest (96% accuracy)
- Support Vector Machine (98% accuracy)
- Logistic Regression (97% accuracy)

Confusion matrices and detailed performance metrics for each model are available in the `confusion_matrices` directory.

## Warning
This application is for educational purposes only and should not be used as a substitute for professional medical diagnosis.

## Version Notes
- The model was trained using scikit-learn version 1.2.2
- Compatible with newer versions but will show version warnings
- Tested on Python 3.8+ and Windows 10/11
