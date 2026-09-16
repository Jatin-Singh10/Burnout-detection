# 🧠 AI-Based Early Burnout Detection System

An intelligent, early-warning decision-support system designed to estimate the risk of burnout among students. This project leverages a calibrated machine learning model to analyze academic, lifestyle, and psychosocial indicators, providing users with an instant risk assessment and personalized, categorized recommendations.

## 🚀 Key Features

- **Interactive Streamlit Dashboard**: A user-friendly interface with tabbed input sections for reduced cognitive load.
- **AI-Powered Risk Prediction**: Uses a trained and calibrated Logistic Regression model to calculate a burnout risk probability.
- **Categorized Recommendation Engine**: Provides tailored advice across four key dimensions:
    - 🚨 **Immediate Support**: High-priority alerts for critical risks.
    - 📚 **Academic Strategies**: Tips for workload and study management.
    - 🌿 **Wellness & Lifestyle**: Guidance on sleep, diet, and balance.
    - 💼 **Professional Guidance**: Pointers toward financial and mental health support.
- **Explainable Logic**: Incorporates "Domain Features" (like Sleep-Work Imbalance and Academic Strain) to capture complex burnout patterns.
- **Critical Alert System**: Triggers immediate emergency warnings for critical indicators (e.g., suicidal thoughts) regardless of the probability score.

## 🛠️ Technical Architecture

### 1. The ML Model
- **Algorithm**: Logistic Regression (selected for its transparency and performance on the specific dataset).
- **Calibration**: Platt-calibrated to ensure the predicted probabilities are reliable.
- **Optimization**: Tuned for **High Recall** to prioritize catching at-risk individuals early.
- **Performance**: High ROC-AUC and PR-AUC, ensuring a strong balance between sensitivity and specificity.

### 2. Risk Assessment Logic
The system categorizes risk based on the calculated probability:

| Risk Tier | Probability Range | Action/Meaning |
| :--- | :--- | :--- |
| **Low** | $0\% - 34\%$ | Generally healthy balance; maintain current habits. |
| **Medium** | $35\% - 64\%$ | Moderate risk; monitor indicators and implement self-care. |
| **High** | $65\% - 100\%$ | High risk; strongly recommend professional consultation. |

**Early-Warning Trigger:** A specific threshold (defined in `metadata.json`) triggers a high-visibility alert when exceeded.

### 3. Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **ML Library**: Scikit-Learn, Joblib
- **Data Handling**: Pandas, NumPy
- **Deployment**: Docker, Streamlit Community Cloud, Hugging Face Spaces

## 📦 Installation & Usage

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Jatin-Singh10/Burnout-detection.git
   cd Burnout-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

### Deployment
The project is ready for deployment via:
- **Streamlit Community Cloud**: Connect your GitHub repo to `share.streamlit.io`.
- **Hugging Face Spaces**: Create a new Space with the Streamlit SDK.
- **Docker**:
  ```bash
  docker build -t burnout-app .
  docker run -p 8501:8501 burnout-app
  ```

## ⚠️ Disclaimer
This system is a **research prototype** and a decision-support tool. It is **NOT** a clinical diagnostic tool or a substitute for professional mental health assessment. If you or someone you know is in crisis, please contact a qualified counselor or a local mental health helpline immediately.
