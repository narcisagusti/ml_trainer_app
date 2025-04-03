# Interactive Machine Learning Model Trainer

## Overview

This project is a web application built with Streamlit that provides an interactive interface for training and evaluating basic machine learning models. Users can easily:

*   Load standard datasets from Seaborn or upload their own CSV files.
*   Visually inspect the data.
*   Select target and feature variables.
*   Choose between different regression or classification models.
*   Configure basic model hyperparameters.
*   Train the model with a single click.
*   Evaluate the model using relevant performance metrics.
*   Visualize the results (e.g., residuals, confusion matrix, ROC curve, feature importance).
*   Download the trained model pipeline for later use.

The goal is to provide a simple, user-friendly tool for quick ML model exploration and experimentation without writing extensive code for each step.

## How the Code Works (Procedure)

The application follows a structured procedure, leveraging several key Python libraries:

1.  Frontend & Control Flow (Streamlit):
    *   The user interface is built entirely using Streamlit widgets (`st.selectbox`,  `st.button`, `st.dataframe`, `st.pyplot`...).
    *   Streamlit's **Session State** (`st.session_state`) is used extensively to store user selections, loaded data, the trained model, and results, ensuring persistence across user interactions.
    *   **Forms** (`st.form`) group configuration widgets in the sidebar. Model training is only triggered when the "Fit Model" button inside the form is clicked, preventing unnecessary reruns.
    *   **Caching** (`@st.cache_data`) is used for data loading functions (`load_seaborn_dataset`, `load_uploaded_file`) to improve performance by avoiding redundant data loading.

2.  Data Handling (Pandas & Seaborn):
    *   `Seaborn` is used to load standard example datasets.
    *   `Pandas` is used for loading uploaded CSVs (`pd.read_csv`) and for all internal data manipulation (DataFrame creation, column selection, handling NaNs, checking dtypes).

3.  Machine Learning Workflow (Scikit-learn):
    *   **Task Identification:** The code attempts to automatically determine if the task is 'Regression' or 'Classification' based on the selected target variable's data type and number of unique values.
    *   **Preprocessing:**
        *   A `ColumnTransformer` is used to apply different preprocessing steps to numerical and categorical features separately.
        *   Numerical features are imputed (missing values filled with the mean using `SimpleImputer`) and scaled (`StandardScaler`).
        *   Categorical features are imputed (missing values filled with the most frequent value using `SimpleImputer`) and then converted into numerical format using `OneHotEncoder`.
        *   Target variables for classification tasks are encoded using `LabelEncoder` if they are not already numeric.
    *   **Modeling:**
        *   Based on the detected task type, appropriate models (`LinearRegression`, `RandomForestRegressor` for Regression; `LogisticRegression`, `RandomForestClassifier` for Classification) are presented to the user.
        *   The selected Scikit-learn model is instantiated with user-defined hyperparameters.
    *   **Pipeline:** The preprocessing steps (`ColumnTransformer`) and the chosen model are chained together into a single Scikit-learn `Pipeline`. This ensures that the same preprocessing steps applied during training are automatically applied during prediction and evaluation, preventing data leakage and simplifying the workflow.
    *   **Training & Prediction:** The `pipeline.fit(X_train, y_train)` method trains the entire sequence. `pipeline.predict()` and `pipeline.predict_proba()` are used on the test set.
    *   **Evaluation:** Scikit-learn's `metrics` module is used to calculate performance scores (MSE, Accuracy, Confusion Matrix, ROC, etc).
    *   **Multiclass ROC:** For classification, the code handles both binary and multiclass ROC curve generation using the One-vs-Rest (OvR) strategy via `label_binarize` and iterative plotting.

4.  Visualization (Matplotlib & Seaborn):
    *   `Matplotlib` serves as the base plotting library.
    *   `Seaborn` is used for some plots like histograms (`sns.histplot`) and heatmaps (`sns.heatmap` for confusion matrix).
    *   Specific plots (Residuals, Confusion Matrix, ROC Curve, Feature Importance) are generated based on the task type and model results.
    *   The generated plots are displayed in the Streamlit app using `st.pyplot()`.

5.  Model Persistence (Joblib):
    *   The entire trained Scikit-learn `Pipeline` object (which includes the fitted preprocessor and model) along with the `LabelEncoder` (if used) is saved into a binary file using `joblib.dump`.
    *   This file is provided to the user for download via `st.download_button`, allowing them to load and reuse the exact trained pipeline later.

## How the Code Was Achieved (Development Process)

This application was developed iteratively based on a set of requirements for an interactive ML trainer:

1.  CORE FUNCTIONALITY: The initial focus was on setting up the Streamlit interface, loading data (Seaborn/CSV), basic feature/target selection, and implementing at least one simple model (e.g., Linear Regression).
2.  ADDING FEATURES: Multiple models (Random Forest), hyperparameter tuning options, and basic performance metrics were added.
3.  IMPROVING UX: Streamlit Forms (`st.form`) were implemented to prevent the model from retraining on every widget change, requiring an explicit "Fit Model" button press. Session State (`st.session_state`) was used to maintain the application's state and results. Caching (`st.cache_data`) was added for data loading.
4.  ENHANCING EVALUATION: Advanced visualizations were incorporated: confusion matrix, ROC curve, feature importance plots. This involved handling specifics like extracting feature names after transformation and calculating probabilities.
5.  ERROR HANDLING: Through testing and feedback (like the errors encountered during our interaction), several issues were identified and fixed:
    *   Correctly determining classification vs. regression tasks.
    *   Handling multiclass ROC curve plotting (implementing OvR).
    *   Resolving index mismatch errors after dropping NaNs by ensuring proper index resets.
    *   Addressing variable scope issues (`NameError` for `submitted`).
    *   Refining the user interface flow (e.g., removing placeholder options, ensuring default selections).
6.  ADVANCED FEATURE: Custom dataset upload and model export (`joblib`) were added to fulfill higher-level requirements.
7.  CODE STRUCTURE: Helper functions were created for clarity and reusability (e.g., `load_data`, `determine_task_type`). Standard libraries were prioritized.

The final code represents a refined version that addresses the initial requirements and incorporates fixes and improvements based on testing and identified errors.

