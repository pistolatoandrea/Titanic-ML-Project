# 🚢 Titanic ML Project

### 🚀 **[View the Live Interactive Demo]https://titanic-ml-project.streamlit.app/**

This project is an end-to-end data science analysis that predicts passenger survival from the Titanic disaster.

All detailed analysis, exploratory data analysis (EDA), and model comparison are located within the main Jupyter Notebook 

### Project Quick Links

* **Full Analysis Notebook:** `[Titanic_Survival_Analysis.ipynb](Titanic_Survival_Analysis.ipynb)`
* **Interactive App Code:** `[app.py](app.py)`

### Tech Stack

* Python
* Pandas (for data cleaning and manipulation)
* Scikit-learn (for model training and evaluation)
* Streamlit (for the interactive demo)
* Joblib (for saving the trained model)

### Deployment Process

The interactive demo is hosted on **Streamlit Cloud**. The process was:

1.  **Model Export:** The final, tuned Random Forest model (`.pkl`) was saved from the notebook.
2.  **App Creation:** An `app_titanic.py` script was built to load the model and generate a UI with `Streamlit`.
3.  **GitHub Push:** The app (`app_titanic.py`), the model (`.pkl`), and dependencies (`requirements.txt`) were pushed to this repository.
4.  **Deploy:** The repository was linked to Streamlit Cloud, which automatically built and published the app at the link above.
