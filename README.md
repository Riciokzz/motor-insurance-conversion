
# Motor Insurance Conversion Modelling

This project aims to predict the likelihood of a customer converting a motor insurance quote into an actual policy, and uncover key insights to support marketing and pricing strategy optimization.

---

## Project Structure

```
.
├── data/                   # Input datasets (CSV and Parquet)
│   ├── conversion_data.csv
│   ├── conversion_data.parquet
│   ├── vehicle_classifier.csv
│   └── vehicle_classifier.parquet
├── functions/             # Local modules
│   ├── machine_learning_models.py
│   ├── plot_utils.py
│   └── utils.py
├── models/
│   ├── final_best_model.pkl
│   └── results_df.csv     # Models comparison results
├── Motor Insurance Conversion Modelling.ipynb
└── requirements.txt       # Project dependencies
```

---

## ⚙Installation

1. **Go to project folder**:
   ```bash
   cd MotorInsurance
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # On Windows
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

1. Launch **Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. Open and run:
   `Motor Insurance Conversion Modelling.ipynb`

   This notebook includes:
   - Data loading and merging
   - Exploratory Data Analysis
   - Feature engineering and correlation analysis
   - Model training with hyperparameter tuning
   - Evaluation and visualization of model performance
   - Business insights and recommendations

---

## Deliverables

- Jupyter Notebook with full pipeline
- `results_df.csv`: Performance comparison of all trained models
- `Task.docx`: Problem statement and context
- Short summary and insights in the final notebook

---
