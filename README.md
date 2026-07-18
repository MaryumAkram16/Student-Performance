# Student Performance Predictor

A Streamlit web app that predicts student academic performance using a machine learning model trained with scikit-learn.

## 🚀 Live App

(https://student-performance-index-prediction.streamlit.app/)

## 📋 Features

- Predicts student performance based on input features
- Interactive, easy-to-use web interface built with Streamlit
- Visualizations powered by Matplotlib and Seaborn

## 🛠 Tech Stack

- **Streamlit** — web app framework
- **scikit-learn** — machine learning model
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — data visualization
- **joblib** — model serialization

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/MaryumAkram16/student-performance-api.git
cd student-performance-api
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the app locally:

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

## 📁 Project Structure

```
student-performance-api/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── runtime.txt          # Pinned Python version for deployment
├── .gitignore
└── README.md
```

## 🧠 Model

[Briefly describe your model here — e.g. algorithm used, input features, target variable, and dataset source]

## 📄 Requirements

```
streamlit==1.59.1
scikit-learn==1.6.1
joblib==1.5.3
numpy==1.26.4
pandas
seaborn
matplotlib
```

## 📌 Deployment Notes

This app is deployed via Streamlit Cloud. `runtime.txt` pins the Python version to `3.11` to ensure prebuilt wheels are available for all dependencies (avoids build failures for packages like `pillow` and `scikit-learn` on newer Python versions).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/MaryumAkram16/student-performance-api/issues).

## 📝 License

[Add your license here, e.g. MIT]
