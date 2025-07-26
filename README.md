# ACM 30-Day Machine Learning Challenge – Joshitha Nambakam
# Cycle1

---

## Daily Progress

| Day | Task Summary |
|-----|--------------|
| *Day 1* | Performed basic data cleaning and exploratory data analysis (EDA) on a burnout dataset by handling the  missing values,fixing the outliners,datavisualization . |
| *Day 2* | Applied preprocessing techniques for machine learning. Encoded categorical variables using OneHotEncoder,OrdinalEncoder and combined them with scaled numerical features to prepare the dataset for modeling. Performed Regression like Linear,Lasso and Ridge to decide which is best among three the types. |
| *Day 3* | Trained classification models using Logistic Regression and Linear Discriminant Analysis (LDA). Evaluated their performance using accuracy, confusion matrix, and ROC-AUC score. Additionally, plotted the ROC curves to visually compare both models’ classification capabilities. |
| *Day 4* | Trained Decision Tree, Random Forest, and k-NearestNeighbors models. Used RandomForest to pick top 3 features and compared model accuracy before and after feature selection.|
| *Day 5* | Trained using Random Forest, burnout risk was predicted after preprocessing with encoding and scaling. Top 3 important features were selected using feature importance, and a minimal model was built using them, achieving good accuracy. A heatmap was plotted to show feature correlations.
| *Main Challenge *|A regression model was developed to predict medical insurance charges. The dataset was preprocessed using label encoding and scaling techniques. A Random Forest Regressor was trained on the full feature set, achieving a high R² score. Using feature importance, the top three influential features were identified. A simpler model using only these top three features was then trained, which maintained nearly the same accuracy as the full model while being more efficient.|
---

## Repository Contents

- Day1.ipynb – Data cleaning and exploration
- Day2.ipynb – Feature encoding and preprocessing
- Day3.ipynb - Classifier Arena
- Day4.ipynb - Tree-Based Models + k-NN + Feature Selection
- Day5.ipynb - 3-Feature Showdown
- Main Challenge - Medical cost regression 
---
# Cycle1
---

## Daily Progress

| Day | Task Summary |
|-----|--------------|
| *Day 6* |Trained Random Forest, AdaBoost, and XGBoost on the Breast Cancer Dataset,performed basic error analysis using classification report  |
| *Day 7* |Applied SVM classification on the Credit Card dataset using Linear, RBF, and Polynomial kernels. Applied PCA for 2D visualization. Compared models using accuracy.|
| *Day 8* |Worked on Unsupervising using clustering methods like K-means and hierarchical and applied PCA and t-SNE for visualization .|
| *Day 9* |Used TF-IDF to transform text data and used SVD for dimensionality reduction ,visualized the first two components and applied KMeans clustering and found Silhouette score.|
| *Day 10* |Applied model validation on the Breast Cancer dataset using Random Forest. Applied K-Fold Cross-Validation to ensure robust performance evaluation. Analyzed the bias-variance trade-off through learning curves, revealing insights into model generalization and potential overfitting/underfitting.|
| *Main Challenge* | Performed  sentiment analysis on the Sentiment140 dataset to classify tweets into Negative, Neutral, and Positive categories. Performed text cleaning, label mapping, and TF-IDF vectorization for feature extraction. Trained a classification model, evaluated it using accuracy, confusion matrix, and classification report along with a heatmap. |
---
## Repository Contents

- Phase1.ipynb – Bagging vs Boosting
- Phase2.ipynb - Support Vector Machines (SVM)
- Phase3.ipynb -  Unsupervised Learning
- Phase4.ipynb -  SVD + PCA
- Phase5.ipynb - Model Validation & Selection
- MainChallenge - Tweet Sentiment Analysis

---
## Tools & Libraries Used
- Python
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- Matplotlib / Seaborn (for visualization, where applicable)
---
