# 🚢 Titanic Survival Prediction: Machine Learning Pet Project

## 📝 Overview
This pet project predicts passenger survival on the Titanic using machine learning techniques.  
The dataset is sourced from **Kaggle's Titanic competition**.  
The project demonstrates skills in **data analysis, feature engineering, and model building** using Python, Pandas, Seaborn, Matplotlib, and Scikit-learn.  

**Goal:** Create a portfolio-worthy project showcasing an end-to-end machine learning workflow.

---

## 📂 Project Structure
/data/ → train.csv, test.csv, gender_submission.csv
/notebooks/ → Jupyter notebooks (e.g., main.ipynb)
/figures/ → Visualization outputs (histograms, bar plots, etc.)
/src/ → Python scripts (planned)

.gitignore → ignores venv/, pycache/, *.ipynb_checkpoints/


---

## 📅 Progress

### 🛠️ Day 1: Project Setup (October 1, 2025)
**Tasks Completed:**
- Created virtual environment (`venv`) to isolate dependencies.  
- Installed libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`.  
- Initialized Git repository (`Titanic-Survival-Prediction`) and pushed to GitHub.  
- Set up project structure: `/data/`, `/notebooks/`, `/figures/`, `/src/`.  
- Added `train.csv`, `test.csv`, and `gender_submission.csv` to `/data/`.  
- Created initial `README.md` and `.gitignore`.  

**Notes:** Environment fully configured, ready for data analysis. No issues encountered.

---

### 📊 Day 2: Data Loading and Initial Visualizations (October 2, 2025)
**Tasks Completed:**
- Loaded datasets in `main.ipynb` using Pandas:
  - `train.csv`: 891 rows, 12 columns.  
  - `test.csv`: 418 rows, 11 columns (no `Survived`).  
  - `gender_submission.csv`: 418 rows, 2 columns (baseline: females survive, males don’t).  

- Inspected data:
  - Missing values: `Age` (177), `Cabin` (687), `Embarked` (2), `Fare` (1).  
  - No duplicates in train/test datasets.  

- Created visualizations:
  - **Age Distribution** → Histogram with KDE (`bins=20`, mean ~29.7, skew ~0.389).  
  - **Survival by Sex** → Females survived more often.  
  - **Survival by Pclass** → 1st class passengers had higher survival rates.  
  - **Correlation Matrix**:
    - `Pclass` vs `Survived`: ~ -0.34  
    - `Fare` vs `Survived`: ~ +0.26  
    - `Age` vs `Survived`: weak correlation (~ -0.08).  

- Saved visualizations to `/figures/`.

**Commit:** `"Day 2: Data loading, initial inspection, and basic visualizations"`  

**Insights:**
- Female passengers had significantly higher survival rates (matches baseline).  
- 1st class passengers survived more often than 3rd class.  
- Missing values in `Age` and `Cabin` require preprocessing.  
- `Pclass` and `Fare` look like strong predictors.

---

## 📈 Visualizations
- Age Distribution
  <img width="1000" height="600" alt="age_distribution" src="https://github.com/user-attachments/assets/1eb8b830-96f1-4c0f-bdfb-cb5c5f84ed38" />
  
- Survival by Sex
  <img width="2098" height="1445" alt="survival_by_sex" src="https://github.com/user-attachments/assets/808aa8f6-498c-42e0-a185-c3bd50afb69d" />

- Survival by Pclass
<img width="800" height="500" alt="survival_by_class" src="https://github.com/user-attachments/assets/5c6b904c-87f0-40c7-863a-37974e308dda" />

  
- Correlation Matrix
<img width="2329" height="2074" alt="correlation_matrix" src="https://github.com/user-attachments/assets/c5e1b9d8-582c-491c-87cf-cf6d71c061a1" />


---

## 🔜 Next Steps
- **Days 3–5:** In-depth EDA (features: `SibSp`, `Parch`, `Embarked`, `Fare`) + missing value handling.  
- **Days 6–7:** Feature engineering (categorical encoding, imputation).  
- **Days 8–10:** Train ML models (logistic regression, random forest).  
- **Days 11–12:** Optimize models & evaluate performance.  
- **Days 13–14:** Finalize documentation & deploy to GitHub.  

---

## 🛠️ Tools Used
- **Python** → Data analysis & modeling  
- **Pandas & NumPy** → Data manipulation  
- **Seaborn & Matplotlib** → Visualizations  
- **Jupyter Notebook** → Interactive analysis  
- **Git & GitHub** → Version control & hosting  
