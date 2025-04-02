import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import label_binarize
import io  # For describe output, model download
import joblib  # For model saving/loading
import traceback # For detailed error reporting

# --- Page Configuration ---
st.set_page_config(page_title="ML Model Trainer", layout="wide", initial_sidebar_state="expanded")

# --- Session State Initialization ---
def init_session_state():
    """Initializes or resets the session state variables."""
    defaults = {
        'data': None,
        'uploaded_file_content': None, # To track if uploaded file changes
        'model_trained': False,
        'model': None,
        'pipeline': None,
        'results': None,
        'feature_importance': None,
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'y_pred': None, 'y_proba': None,
        'feature_names_processed': None,
        'target_encoder': None, # Stores the fitted LabelEncoder for the target
        'trained_model_name': None, # Stores the name of the trained model for download filename
        'selected_target': None, # Store selected target to handle potential reruns/updates
        'selected_features': None, # Store selected features
        'model_type': 'Regression', # Default model type guess
        'numeric_cols': [],
        'categorical_cols': [],
        'available_targets': [],
        'all_features': [],
        'numeric_features_options': [],
        'categorical_features_options': [],
        'last_data_source_type': None,
        'last_data_source_value': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Ensure session state is initialized at the start
if 'initialized' not in st.session_state:
    init_session_state()
    st.session_state.initialized = True


# --- Helper Functions ---
@st.cache_data # Cache data loading
def load_data(source_type, source_value):
    """Loads data from Seaborn or an uploaded CSV file."""
    try:
        if source_type == "Seaborn":
            data = sns.load_dataset(source_value)
            st.success(f"Loaded '{source_value}' dataset from Seaborn.")
            return data
        elif source_type == "Upload" and source_value is not None:
            # Important: Use BytesIO or getvalue() to read the file content
            try:
                 # Attempt to read with default comma delimiter
                 data = pd.read_csv(source_value)
            except Exception as e1:
                 st.warning(f"Failed to read CSV with comma delimiter ({e1}). Trying semicolon...")
                 try:
                     # Reset stream position before trying again
                     if hasattr(source_value, 'seek'):
                         source_value.seek(0)
                     data = pd.read_csv(source_value, sep=';')
                 except Exception as e2:
                     st.error(f"Failed to read CSV with semicolon delimiter as well ({e2}). Please ensure it's a valid CSV.")
                     return None
            st.success("CSV file uploaded successfully.")
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_column_types(df):
    """Identifies numeric and categorical columns in a DataFrame."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numeric_cols, categorical_cols

def update_data_dependent_state(df):
    """Updates session state variables that depend on the loaded data."""
    st.session_state.numeric_cols, st.session_state.categorical_cols = get_column_types(df)
    st.session_state.available_targets = df.columns.tolist()
    st.session_state.all_features = df.columns.tolist() # Initially, all columns are potential features
    # Reset selections if data changes significantly
    st.session_state.selected_target = None
    st.session_state.selected_features = None
    st.session_state.model_trained = False # New data means model isn't valid anymore
    st.session_state.results = None
    st.session_state.feature_importance = None
    st.session_state.model = None
    st.session_state.pipeline = None


# --- Sidebar for Data Input ---
with st.sidebar:
    st.header("1. Data Input")
    data_option = st.radio("Choose data source:", ["Seaborn Dataset", "Upload CSV"], key="data_source", horizontal=True)

    current_data_source_type = None
    current_data_source_value = None

    if data_option == "Seaborn Dataset":
        # Removed 'mpg' and 'car_crashes'
        dataset_names = ["tips", "iris", "diamonds", "planets", "titanic", "penguins"]
        # Set a valid default (penguins is still in the list)
        default_idx = 0 # Default to first item if penguins somehow missing
        if "penguins" in dataset_names:
            default_idx = dataset_names.index("penguins")
        selected_dataset = st.selectbox("Select dataset:", dataset_names, index=default_idx)
        current_data_source_type = "Seaborn"
        current_data_source_value = selected_dataset
    else: # Upload CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        current_data_source_type = "Upload"
        current_data_source_value = uploaded_file

    # Check if data source has changed or if new file uploaded
    data_changed = False
    if current_data_source_type != st.session_state.get('last_data_source_type') or \
       (current_data_source_type == "Seaborn" and current_data_source_value != st.session_state.get('last_data_source_value')) or \
       (current_data_source_type == "Upload" and uploaded_file is not None and uploaded_file.getvalue() != st.session_state.get('uploaded_file_content')):

        data_changed = True
        st.session_state.data = load_data(current_data_source_type, current_data_source_value)
        st.session_state.last_data_source_type = current_data_source_type
        st.session_state.last_data_source_value = current_data_source_value
        if current_data_source_type == "Upload" and uploaded_file is not None:
            st.session_state.uploaded_file_content = uploaded_file.getvalue() # Store content hash/value
        else:
            st.session_state.uploaded_file_content = None

        if st.session_state.data is not None:
            update_data_dependent_state(st.session_state.data) # Update column lists etc.
        else:
            init_session_state() # Reset completely if loading failed


# --- Main Area ---
st.title("🚀 ML Model Trainer")
st.write("Load data via the sidebar, configure your model below, train, and analyze the results.")

if st.session_state.data is None:
    st.info("👈 Please load data using the sidebar to begin.")
else:
    df_display = st.session_state.data

    # --- Data Preview and EDA ---
    st.header("📊 Data Preview & Exploration")
    with st.expander("Show Dataset Overview", expanded=False):
        st.dataframe(df_display.head())
        buffer = io.StringIO()
        df_display.info(buf=buffer)
        st.text(f"Dataset Shape: {df_display.shape}")
        st.text("Column Info:")
        st.text(buffer.getvalue())

        st.subheader("Basic Statistics")
        try:
             st.dataframe(df_display.describe(include='all'))
        except Exception as e:
             st.warning(f"Could not generate descriptive statistics: {e}")

    with st.expander("Explore Data Visually (EDA)", expanded=False):
        st.subheader("Distributions")
        plot_type = st.radio("Select plot type:", ["Histogram (Numeric)", "Count Plot (Categorical)"], horizontal=True, key="eda_plot_type")

        if plot_type == "Histogram (Numeric)":
             numeric_cols_eda = st.session_state.numeric_cols
             if numeric_cols_eda:
                 col_to_plot_hist = st.selectbox("Select numeric column for histogram:", numeric_cols_eda, key="hist_select")
                 if col_to_plot_hist:
                     fig_hist, ax_hist = plt.subplots()
                     try:
                         sns.histplot(df_display[col_to_plot_hist].dropna(), kde=True, ax=ax_hist) # Drop NA for plotting
                         ax_hist.set_title(f'Distribution of {col_to_plot_hist}')
                         st.pyplot(fig_hist)
                         plt.close(fig_hist) # Close figure
                     except Exception as e:
                         st.warning(f"Could not plot histogram for {col_to_plot_hist}: {e}")
             else:
                 st.info("No numeric columns available for histograms.")

        else: # Count Plot
            cat_cols_eda = st.session_state.categorical_cols
            if cat_cols_eda:
                col_to_plot_count = st.selectbox("Select categorical column for count plot:", cat_cols_eda, key="count_select")
                if col_to_plot_count:
                    fig_count, ax_count = plt.subplots()
                    try:
                        # Limit categories shown if too many
                        top_n = 20
                        if df_display[col_to_plot_count].nunique() > top_n:
                             st.info(f"Showing top {top_n} categories for '{col_to_plot_count}'.")
                             top_categories = df_display[col_to_plot_count].value_counts().nlargest(top_n).index
                             df_plot = df_display[df_display[col_to_plot_count].isin(top_categories)]
                             order = top_categories
                        else:
                            df_plot = df_display
                            order = df_plot[col_to_plot_count].value_counts().index

                        sns.countplot(data=df_plot, y=col_to_plot_count, ax=ax_count, order=order, palette="viridis") # Use y for better readability
                        ax_count.set_title(f'Counts of {col_to_plot_count}')
                        st.pyplot(fig_count)
                        plt.close(fig_count) # Close figure
                    except Exception as e:
                         st.warning(f"Could not plot countplot for {col_to_plot_count}: {e}")

            else:
                 st.info("No categorical columns available for count plots.")

        st.subheader("Correlations (Numeric Features)")
        numeric_cols_corr = st.session_state.numeric_cols
        if len(numeric_cols_corr) > 1:
            try:
                corr_matrix = df_display[numeric_cols_corr].corr()
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
                ax_corr.set_title('Correlation Matrix of Numeric Features')
                st.pyplot(fig_corr)
                plt.close(fig_corr) # Close figure
            except Exception as e:
                st.warning(f"Could not compute or plot correlation matrix: {e}")
        else:
            st.info("Need at least two numeric columns to compute correlations.")

    st.divider()

    # --- Model Configuration Form ---
    st.header("⚙️ Configure Model Training")

    # Using a form to gather all configuration before running the training
    with st.form(key="training_form"):
        col1, col2 = st.columns([2, 3]) # Adjust column widths as needed

        with col1:
            st.subheader("Target & Model Type")
            if not st.session_state.available_targets:
                st.warning("No columns found in the dataset.")
                target_variable = None
            else:
                # Sensible default: last column or a common target name if found
                default_target_index = len(st.session_state.available_targets) - 1
                common_targets = ['target', 'class', 'species', 'price', 'survived', 'quality', 'charges', 'result']
                for t in common_targets:
                    if t in st.session_state.available_targets:
                        default_target_index = st.session_state.available_targets.index(t)
                        break

                # Ensure default index is valid
                if default_target_index >= len(st.session_state.available_targets):
                     default_target_index = 0

                # Use session state to remember the selection if possible
                current_target_selection_index = default_target_index
                if st.session_state.selected_target in st.session_state.available_targets:
                    current_target_selection_index = st.session_state.available_targets.index(st.session_state.selected_target)

                target_variable = st.selectbox(
                    "Select Target Variable (y):",
                    st.session_state.available_targets,
                    index=current_target_selection_index,
                    key='target_var_select_form'
                )
                st.session_state.selected_target = target_variable # Store the selection


                if target_variable:
                    # Automatically guess model type based on target dtype and unique values
                    target_dtype = df_display[target_variable].dtype
                    unique_vals = df_display[target_variable].nunique()

                    # Update initial guess if target changed
                    if pd.api.types.is_numeric_dtype(target_dtype) and unique_vals > 15: # Heuristic for regression
                        st.session_state.model_type = "Regression"
                    elif pd.api.types.is_numeric_dtype(target_dtype) and unique_vals <= 2 : # Likely Binary
                         st.session_state.model_type = "Classification"
                    elif not pd.api.types.is_numeric_dtype(target_dtype): # Categorical/Object
                         st.session_state.model_type = "Classification"
                    else: # Numeric with few unique values - could be either, default to Classification
                        st.session_state.model_type = "Classification"


                    model_type_selected = st.radio(
                        "Select Model Type:",
                        ["Regression", "Classification"],
                        index=["Regression", "Classification"].index(st.session_state.model_type), # Use state for default
                        key='model_type_radio_form',
                        horizontal=True
                    )
                    st.session_state.model_type = model_type_selected # Update state based on user choice

                    # Update available features based on selected target
                    st.session_state.all_features = [col for col in df_display.columns if col != target_variable]
                    st.session_state.numeric_features_options = [col for col in st.session_state.numeric_cols if col != target_variable]
                    st.session_state.categorical_features_options = [col for col in st.session_state.categorical_cols if col != target_variable]

            st.subheader("Train-Test Split")
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key='test_size_form')
            random_state = st.number_input("Random State", 0, 1000, 42, 1, key='random_state_form')


        with col2:
            st.subheader("Feature Selection (X)")
            if target_variable and st.session_state.all_features: # Only show if target is selected
                select_all = st.checkbox("Select All Features", value=True, key="select_all_features")
                if select_all:
                    selected_features = st.session_state.all_features
                    st.multiselect(
                        "Selected Features:",
                        st.session_state.all_features,
                        default=st.session_state.all_features,
                        key='features_select_form_disabled', # Different key needed
                        disabled=True
                    )
                else:
                    # Remember previous selection if available and still valid
                    default_selection = st.session_state.all_features
                    if st.session_state.selected_features:
                         # Filter previous selection to ensure they are still valid features
                         valid_previous = [f for f in st.session_state.selected_features if f in st.session_state.all_features]
                         if valid_previous:
                              default_selection = valid_previous

                    selected_features = st.multiselect(
                        "Select Features:",
                        st.session_state.all_features,
                        default=default_selection,
                        key='features_select_form' # Original key
                     )
                st.session_state.selected_features = selected_features # Store selection
            else:
                st.warning("Select a target variable first to choose features.")
                selected_features = []


            # --- Preprocessing Options ---
            st.subheader("Preprocessing")
            with st.expander("Configure Preprocessing Steps", expanded=False):
                st.write("**Numeric Features:**")
                num_imputation = st.selectbox("Missing Value Imputation:", ["mean", "median"], key='num_impute_form')
                scale_numeric = st.checkbox("Scale Numeric Features (StandardScaler)", value=True, key='scale_num_form')

                st.write("**Categorical Features:**")
                cat_imputation = st.selectbox("Missing Value Imputation:", ["most_frequent", "constant"], key='cat_impute_form')
                cat_impute_fill_value = "missing" # Default
                if cat_imputation == "constant":
                     cat_impute_fill_value = st.text_input("Fill value for 'constant' imputation:", "missing", key='cat_fill_val_form')


        # --- Model Selection & Parameters (Outside columns, spans full width within form) ---
        st.subheader("Model Selection & Hyperparameters")
        model_option = None # Initialize

        if 'model_type' in st.session_state: # Check if model_type is set
            if st.session_state.model_type == "Regression":
                model_option_reg = st.selectbox(
                    "Select Regression Model:",
                    ["Linear Regression", "Random Forest Regressor"],
                    key='reg_model_form'
                )
                model_option = model_option_reg # Assign to generic variable
                if model_option_reg == "Random Forest Regressor":
                    with st.expander("Random Forest Hyperparameters", expanded=True):
                        rf_n_estimators_reg = st.slider("Number of Estimators", 10, 500, 100, 10, key='rf_n_est_reg_form')
                        rf_max_depth_reg = st.slider("Maximum Depth", 1, 50, 10, 1, key='rf_max_depth_reg_form')
                        rf_min_samples_split_reg = st.slider("Minimum Samples Split", 2, 20, 2, 1, key='rf_min_split_reg_form')

            elif st.session_state.model_type == "Classification":
                model_option_cls = st.selectbox(
                    "Select Classification Model:",
                    ["Logistic Regression", "Random Forest Classifier"],
                    key='cls_model_form'
                )
                model_option = model_option_cls # Assign to generic variable
                if model_option_cls == "Random Forest Classifier":
                     with st.expander("Random Forest Hyperparameters", expanded=True):
                        rf_n_estimators_cls = st.slider("Number of Estimators", 10, 500, 100, 10, key='rf_n_est_cls_form')
                        rf_max_depth_cls = st.slider("Maximum Depth", 1, 50, 10, 1, key='rf_max_depth_cls_form')
                        rf_min_samples_split_cls = st.slider("Minimum Samples Split", 2, 20, 2, 1, key='rf_min_split_cls_form')
                elif model_option_cls == "Logistic Regression":
                     with st.expander("Logistic Regression Hyperparameters", expanded=True):
                        log_reg_C = st.number_input("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01, key='log_reg_C_form')
                        log_reg_solver = st.selectbox("Solver", ['liblinear', 'lbfgs', 'saga'], key='log_reg_solver_form')


        # --- Form Submit Button ---
        st.divider()
        submitted = st.form_submit_button("🚀 Train Model", type="primary", use_container_width=True)

    # --- Training Logic (Runs only when the form is submitted) ---
    if submitted:
        st.session_state.model_trained = False # Reset trained flag on new submission attempt

        # Access form values using their keys
        target_variable_submit = st.session_state.selected_target # Use state which holds selection
        selected_features_submit = st.session_state.selected_features # Use state
        model_type_submit = st.session_state.model_type # Use state
        test_size_submit = test_size # From slider in form
        random_state_submit = random_state # From number input in form
        num_imputation_submit = num_imputation # From selectbox in form
        scale_numeric_submit = scale_numeric # From checkbox in form
        cat_imputation_submit = cat_imputation # From selectbox in form
        cat_impute_fill_value_submit = cat_impute_fill_value # From text input if 'constant'
        model_option_submit = model_option # The model name selected in the form


        if not selected_features_submit:
            st.error("Please select at least one feature (X).")
        elif not target_variable_submit:
             st.error("Please select a target variable (y).")
        elif not model_option_submit:
             st.error("Please select a model.")
        else:
            with st.spinner("⚙️ Preparing data and training model... Please wait."):
                try:
                    # --- Data Preparation ---
                    df_processed = st.session_state.data.copy()

                    # Drop rows with NaN in target variable BEFORE splitting
                    initial_rows = len(df_processed)
                    df_processed.dropna(subset=[target_variable_submit], inplace=True)
                    rows_after_dropna = len(df_processed)
                    if rows_after_dropna < initial_rows:
                        st.warning(f"Dropped {initial_rows - rows_after_dropna} rows with missing target variable ('{target_variable_submit}').")

                    if df_processed.empty:
                         st.error("No data remaining after dropping rows with missing target values. Cannot proceed.")

                    else:

                        X = df_processed[selected_features_submit]
                        y = df_processed[target_variable_submit]

                        # Identify selected numeric/categorical features *within the submitted selection*
                        selected_numeric_submit = [f for f in selected_features_submit if f in st.session_state.numeric_cols]
                        selected_categorical_submit = [f for f in selected_features_submit if f in st.session_state.categorical_cols]

                        # Handle Target Encoding for Classification
                        st.session_state.target_encoder = None # Reset target encoder
                        if model_type_submit == "Classification":
                            if not pd.api.types.is_numeric_dtype(y):
                                st.session_state.target_encoder = LabelEncoder()
                                y = st.session_state.target_encoder.fit_transform(y)
                                st.info(f"Target variable '{target_variable_submit}' encoded using LabelEncoder. Classes: {list(st.session_state.target_encoder.classes_)}")
                            else:
                                # Check if it looks like regression mistakenly chosen for classification
                                if y.nunique() > 20: # Heuristic
                                     st.warning(f"Target variable '{target_variable_submit}' is numeric with {y.nunique()} unique values. Ensure it's suitable for classification.")
                                # Ensure target is integer type for classification metrics/models if it's float but looks categorical (e.g., 0.0, 1.0)
                                if pd.api.types.is_float_dtype(y) and y.nunique() < 10:
                                    # Attempt conversion if values seem like integers
                                    try:
                                        y = y.astype(int)
                                        st.info(f"Converted float target '{target_variable_submit}' to integers.")
                                    except ValueError:
                                        st.warning(f"Could not convert float target '{target_variable_submit}' to int, potential mixed types?")


                        # Split data FIRST
                        try:
                             # Calculate value counts for the target variable y
                             # Convert to Series to reliably use value_counts, even if y is a numpy array
                             y_series = pd.Series(y)
                             y_counts = y_series.value_counts()

                             # Check if stratification is feasible
                             can_stratify = True
                             # Condition: Classification model AND target has more than 1 unique value AND minimum class count is >= 2
                             if model_type_submit == 'Classification' and len(y_counts) > 1 and y_counts.min() < 2:
                                 can_stratify = False
                                 st.warning(f"⚠️ Stratification disabled: The target variable '{target_variable_submit}' has classes with fewer than 2 samples. Performing non-stratified split.")
                             elif model_type_submit != 'Classification' or len(y_counts) <= 1:
                                 # Also disable stratification if not classification or only one class exists
                                 can_stratify = False

                             # Determine the stratify argument
                             stratify_arg = y if can_stratify else None

                             # Perform split
                             X_train, X_test, y_train, y_test = train_test_split(
                                X, y,
                                test_size=test_size_submit,
                                random_state=random_state_submit,
                                stratify=stratify_arg # Use the determined argument
                             )
                             st.session_state.X_train = X_train
                             st.session_state.X_test = X_test
                             st.session_state.y_train = y_train
                             st.session_state.y_test = y_test

                        except ValueError as e:
                            # This except block now serves as a fallback for *other* potential ValueErrors during split
                            st.error(f"❌ Error during train-test split: {e}")
                            st.warning("Attempting non-stratified split as a fallback...")
                            try:
                                # Fallback explicitly without stratification
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y,
                                    test_size=test_size_submit,
                                    random_state=random_state_submit,
                                    stratify=None # Explicitly disable stratification in fallback
                                )
                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test
                                st.success("✅ Fallback non-stratified split successful.")
                            except Exception as fallback_e:
                                st.error(f"❌ Fallback train-test split also failed: {fallback_e}")
                                # Stop execution here if split fails completely
                                st.session_state.model_trained = False # Ensure model is marked as not trained
                                raise fallback_e # Reraise the error to halt processing

                        # --- Continue with Preprocessing Pipeline Construction ONLY if split was successful ---
                        # (The rest of the training code follows here, OUTSIDE the try...except block for the split)
                        # Ensure X_train, X_test etc. were actually assigned before proceeding

                        if 'X_train' not in st.session_state or st.session_state.X_train is None:
                             st.error("❌ Train-test split failed. Cannot proceed with model training.")
                             # Optional: You might want to exit the 'if submitted:' block here
                             st.stop() # Or handle more gracefully

                        # --- Preprocessing Pipeline Construction ---
                        # (Rest of your preprocessing and model fitting code...)


                        # --- Preprocessing Pipeline Construction ---
                        numeric_transformer_steps = []
                        if selected_numeric_submit and X_train[selected_numeric_submit].isnull().sum().sum() > 0: # Only add imputer if needed and if numeric cols exist
                            numeric_transformer_steps.append(('imputer', SimpleImputer(strategy=num_imputation_submit)))
                        if scale_numeric_submit and selected_numeric_submit: # Only add scaler if selected and numeric cols exist
                            numeric_transformer_steps.append(('scaler', StandardScaler()))
                        # Use 'passthrough' if no steps are needed or no numeric features selected, else create Pipeline
                        numeric_transformer = Pipeline(steps=numeric_transformer_steps) if numeric_transformer_steps else 'passthrough'


                        categorical_transformer_steps = []
                        # Check if there are any categorical features selected *and* if they have missing values
                        if selected_categorical_submit and X_train[selected_categorical_submit].isnull().sum().sum() > 0:
                           cat_imputer = SimpleImputer(
                               strategy=cat_imputation_submit,
                               fill_value=(cat_impute_fill_value_submit if cat_imputation_submit == 'constant' else None)
                           )
                           categorical_transformer_steps.append(('imputer', cat_imputer))
                        # Always add OneHotEncoder if there are categorical features selected
                        if selected_categorical_submit:
                             categorical_transformer_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
                        # Use 'passthrough' if no categorical features selected or no steps needed, else create Pipeline
                        categorical_transformer = Pipeline(steps=categorical_transformer_steps) if categorical_transformer_steps else 'passthrough'


                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', numeric_transformer, selected_numeric_submit),
                                ('cat', categorical_transformer, selected_categorical_submit)
                            ],
                            remainder='drop' # Explicitly drop columns not selected - safer
                        )

                        # --- Model Definition ---
                        # Retrieve hyperparameter values from the form using their keys
                        if model_type_submit == "Regression":
                            if model_option_submit == "Linear Regression":
                                model = LinearRegression()
                            elif model_option_submit == "Random Forest Regressor":
                                model = RandomForestRegressor(
                                    n_estimators=rf_n_estimators_reg, # Use form value
                                    max_depth=rf_max_depth_reg if rf_max_depth_reg > 0 else None, # Allow 0 or None for no limit
                                    min_samples_split=rf_min_samples_split_reg,
                                    random_state=random_state_submit
                                )
                        else: # Classification
                            if model_option_submit == "Logistic Regression":
                                model = LogisticRegression(
                                    C=log_reg_C, # Use form value
                                    solver=log_reg_solver, # Use form value
                                    random_state=random_state_submit,
                                    max_iter=1000 # Increase max_iter for some solvers
                                )
                            elif model_option_submit == "Random Forest Classifier":
                                model = RandomForestClassifier(
                                    n_estimators=rf_n_estimators_cls, # Use form value
                                    max_depth=rf_max_depth_cls if rf_max_depth_cls > 0 else None,
                                    min_samples_split=rf_min_samples_split_cls,
                                    random_state=random_state_submit
                                )

                        # --- Create and Fit Full Pipeline ---
                        st.session_state.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                                    ('model', model)])

                        st.session_state.pipeline.fit(X_train, y_train)

                        # --- Prediction and Evaluation ---
                        st.session_state.y_pred = st.session_state.pipeline.predict(X_test)
                        st.session_state.y_proba = None
                        if model_type_submit == "Classification" and hasattr(st.session_state.pipeline, "predict_proba"):
                            try:
                                st.session_state.y_proba = st.session_state.pipeline.predict_proba(X_test)
                            except Exception as e:
                                st.warning(f"Could not generate prediction probabilities: {e}")


                        # --- Store Results ---
                        results = {}
                        if model_type_submit == "Regression":
                            results = {
                                'MSE': mean_squared_error(y_test, st.session_state.y_pred),
                                'RMSE': np.sqrt(mean_squared_error(y_test, st.session_state.y_pred)),
                                'MAE': mean_absolute_error(y_test, st.session_state.y_pred),
                                'R²': r2_score(y_test, st.session_state.y_pred)
                            }
                        else: # Classification
                            # Adjust average strategy based on number of classes
                            # Use the original y before potential label encoding to check nunique for average strategy
                            num_classes = pd.Series(y).nunique() # Ensure y is treated as Series for nunique
                            avg_strategy = 'weighted' if num_classes > 2 else 'binary'
                            # Determine pos_label for binary case, check if target encoder was used
                            pos_label = 1 # Default positive label for binary
                            if avg_strategy == 'binary' and st.session_state.target_encoder:
                                 # Find the encoded value corresponding to the second class name (usually the positive class)
                                 if len(st.session_state.target_encoder.classes_) > 1:
                                      pos_label = st.session_state.target_encoder.transform([st.session_state.target_encoder.classes_[1]])[0]


                            results = {
                                'Accuracy': accuracy_score(y_test, st.session_state.y_pred),
                                'Precision': precision_score(y_test, st.session_state.y_pred, average=avg_strategy, zero_division=0, pos_label=pos_label if avg_strategy == 'binary' else 1), # pos_label only relevant for binary
                                'Recall': recall_score(y_test, st.session_state.y_pred, average=avg_strategy, zero_division=0, pos_label=pos_label if avg_strategy == 'binary' else 1),
                                'F1 Score': f1_score(y_test, st.session_state.y_pred, average=avg_strategy, zero_division=0, pos_label=pos_label if avg_strategy == 'binary' else 1)
                            }
                        st.session_state.results = results

                        # --- Feature Importance ---
                        feature_names_out = []
                        try:
                            # Get feature names after transformation (handles OneHotEncoding)
                            ct_transformer = st.session_state.pipeline.named_steps['preprocessor']
                            feature_names_out = ct_transformer.get_feature_names_out()
                            st.session_state.feature_names_processed = feature_names_out

                            # Access the final model step for importance/coefficients
                            final_model = st.session_state.pipeline.named_steps['model']

                            importances = None # Initialize
                            importance_type = None # 'Importance' or 'Coefficient'

                            if hasattr(final_model, 'feature_importances_'): # Tree-based models
                                importances = final_model.feature_importances_
                                importance_type = 'Importance'
                            elif hasattr(final_model, 'coef_'): # Linear models
                                # Handle potential multi-class coefficients (shape might be > 1D for LogisticRegression OvR)
                                if final_model.coef_.ndim > 1 and model_type_submit == "Classification":
                                     # Show magnitude of coefficients for the first class for simplicity, or average? Let's use max magnitude across classes.
                                     importances = np.max(np.abs(final_model.coef_), axis=0)
                                     st.info("Displaying maximum absolute coefficient magnitude across classes for feature importance.")
                                else:
                                     # Single output (Regression) or Binary Classification coef_ shape (1, n_features)
                                     importances = np.abs(final_model.coef_.ravel()) # Use ravel() to ensure 1D array

                                importance_type = 'Coefficient Magnitude'

                            if importances is not None and importance_type is not None:
                                 # Ensure feature_names_out and importances have the same length
                                 if len(feature_names_out) == len(importances):
                                      st.session_state.feature_importance = pd.DataFrame({
                                          'Feature': feature_names_out,
                                          importance_type: importances
                                      }).sort_values(by=importance_type, ascending=False)
                                 else:
                                      st.warning(f"Mismatch between number of generated feature names ({len(feature_names_out)}) and importance values ({len(importances)}). Cannot display feature importance.")
                                      st.session_state.feature_importance = None
                            else:
                                 st.session_state.feature_importance = None

                        except AttributeError as ae:
                             # This can happen if 'passthrough' was used for a transformer type
                             st.warning(f"Could not retrieve feature names, potentially due to 'passthrough' transformers or model type. Feature importance unavailable. Error: {ae}")
                             st.session_state.feature_importance = None
                        except Exception as e:
                            st.warning(f"Could not extract feature importance/coefficients: {e}")
                            st.session_state.feature_importance = None


                        st.session_state.model = final_model # Store the fitted model itself
                        st.session_state.model_trained = True
                        st.session_state.trained_model_name = model_option_submit # Store the name for download button
                        st.success(f"✅ {model_option_submit} model trained successfully!")

                except MemoryError:
                    st.error("❌ Memory Error: The dataset or model might be too large for the available memory. Try reducing features, data size, or model complexity (e.g., fewer estimators/depth).")
                    st.session_state.model_trained = False
                except ValueError as ve:
                    st.error(f"❌ ValueError during processing or training: {ve}. Check data types (especially target vs. model type), missing values, and feature/target selections.")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.session_state.model_trained = False
                except KeyError as ke:
                     st.error(f"❌ KeyError: A specified column was not found: {ke}. This might happen if the selected features/target are incorrect or change after selection.")
                     st.error(f"Traceback: {traceback.format_exc()}")
                     st.session_state.model_trained = False
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred during training: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.session_state.model_trained = False

    # --- Model Results Display (Only shown if model is successfully trained) ---
    if st.session_state.model_trained:
        st.header("📈 Model Performance & Results")

        # Metrics
        st.subheader("Performance Metrics")
        if st.session_state.results:
            metrics_df = pd.DataFrame({
                'Metric': list(st.session_state.results.keys()),
                'Value': [f"{v:.4f}" for v in st.session_state.results.values()] # Format values
            })
            st.dataframe(metrics_df)
        else:
             st.info("Metrics are not available.")

        # --- Model Download Button ---
        st.subheader("Download Trained Model")
        # Generate filename using the stored model name and data source info
        model_name_for_file = st.session_state.trained_model_name.replace(" ", "_").lower()
        data_source_name = "uploaded_data" # Default
        if st.session_state.last_data_source_type == "Seaborn":
             data_source_name = st.session_state.last_data_source_value
        elif st.session_state.last_data_source_type == "Upload" and st.session_state.uploaded_file_content is not None:
             # Try to get original filename if available, else use default
             if hasattr(st.session_state.last_data_source_value, 'name'):
                  data_source_name = st.session_state.last_data_source_value.name.split('.')[0] # Get filename without extension
             else:
                 data_source_name = "uploaded_data"

        model_filename = f"trained_{model_name_for_file}_on_{data_source_name}.joblib"

        try:
            # Serialize the *entire pipeline* (preprocessor + model)
            model_bytes = io.BytesIO()
            joblib.dump(st.session_state.pipeline, model_bytes)
            model_bytes.seek(0) # Reset buffer to the beginning

            st.download_button(
                label="💾 Download Model Pipeline (.joblib)",
                data=model_bytes,
                file_name=model_filename,
                mime='application/octet-stream', # Standard for binary files
                help="Download the trained ML pipeline (preprocessing + model) as a .joblib file."
            )
            # st.info(f"Click above to download '{model_filename}'") # Removed redundant info
            st.caption("💡 To load this model later in Python: `import joblib; loaded_pipeline = joblib.load('your_downloaded_file.joblib')`")
            # --- MODIFIED LINE: Commented out the security warning ---
            # st.warning("⚠️ **Security:** Only load .joblib files from trusted sources.")
            # --- END OF MODIFICATION ---

        except Exception as e:
            st.error(f"Error preparing model for download: {e}")
            st.error("Ensure 'joblib' is installed (`pip install joblib`).")


        # --- Detailed Visualizations ---
        st.subheader("Result Visualizations")
        y_test_res = st.session_state.y_test
        y_pred_res = st.session_state.y_pred
        y_proba_res = st.session_state.y_proba
        model_type_res = st.session_state.model_type
        feature_importance_res = st.session_state.feature_importance
        target_encoder_res = st.session_state.target_encoder # Get encoder from state

        # Define tabs based on model type
        tab_titles = ["Performance Plots", "Feature Importance"]
        if model_type_res == "Classification":
             tab_titles.append("Classification Specifics")
        else:
             tab_titles.append("Regression Specifics")

        tab1, tab2, tab3 = st.tabs(tab_titles)


        with tab1:
            st.markdown("##### Performance Plots")
            if model_type_res == "Regression":
                fig_res, ax_res = plt.subplots(1, 2, figsize=(15, 5)) # Adjusted size slightly

                # Actual vs Predicted
                ax_res[0].scatter(y_test_res, y_pred_res, alpha=0.7, edgecolors='k', s=50)
                min_val = min(np.min(y_test_res), np.min(y_pred_res)) # Use np.min for safety
                max_val = max(np.max(y_test_res), np.max(y_pred_res)) # Use np.max for safety
                ax_res[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit") # Diagonal line
                ax_res[0].set_xlabel('Actual Values')
                ax_res[0].set_ylabel('Predicted Values')
                ax_res[0].set_title('Actual vs. Predicted Values')
                ax_res[0].grid(True, linestyle='--', alpha=0.6)
                ax_res[0].legend()

                # Residual Plot
                residuals = y_test_res - y_pred_res
                ax_res[1].scatter(y_pred_res, residuals, alpha=0.7, edgecolors='k', s=50)
                ax_res[1].axhline(y=0, color='r', linestyle='--', lw=2, label="Zero Error")
                ax_res[1].set_xlabel('Predicted Values')
                ax_res[1].set_ylabel('Residuals (Actual - Predicted)')
                ax_res[1].set_title('Residual Plot')
                ax_res[1].grid(True, linestyle='--', alpha=0.6)
                ax_res[1].legend()

                st.pyplot(fig_res)
                plt.close(fig_res) # Close figure

            else: # Classification
                fig_cls, ax_cls = plt.subplots(1, 2, figsize=(16, 6)) # Adjusted size slightly

                # Confusion Matrix
                cm = confusion_matrix(y_test_res, y_pred_res)
                # Get class labels correctly, handling potential LabelEncoder
                cm_labels = 'auto'
                if target_encoder_res:
                     cm_labels = target_encoder_res.classes_

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cls[0],
                                   xticklabels=cm_labels,
                                   yticklabels=cm_labels)
                ax_cls[0].set_xlabel('Predicted Label')
                ax_cls[0].set_ylabel('True Label')
                ax_cls[0].set_title('Confusion Matrix')

                # ROC Curve
                ax_cls[1].plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)') # Baseline
                ax_cls[1].set_xlabel('False Positive Rate (FPR)')
                ax_cls[1].set_ylabel('True Positive Rate (TPR)')
                ax_cls[1].set_title('Receiver Operating Characteristic (ROC) Curve')
                ax_cls[1].grid(True, linestyle='--', alpha=0.6)
                ax_cls[1].set_xlim([-0.05, 1.05])
                ax_cls[1].set_ylim([-0.05, 1.05])

                if y_proba_res is not None:
                    n_classes = y_proba_res.shape[1]
                    if n_classes == 2: # Binary case
                        fpr, tpr, _ = roc_curve(y_test_res, y_proba_res[:, 1]) # Prob of positive class
                        roc_auc = auc(fpr, tpr)
                        ax_cls[1].plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
                    else: # Multiclass case (One-vs-Rest using probabilities)
                        # CORRECTED APPROACH using label_binarize

                        # Determine the full set of classes expected by the model for binarization
                        all_classes = None
                        # y_test_res should contain the numeric labels (0 to k-1) at this point
                        # Either because LabelEncoder was used before split, or they were already numeric
                        if target_encoder_res:
                             # Get classes from the encoder used during training prep (provides correct order)
                             all_classes = target_encoder_res.classes_ # These are the original labels
                             # We need the *encoded* version (0 to k-1) for label_binarize's classes parameter
                             numeric_classes = target_encoder_res.transform(all_classes)
                        else:
                             # If no encoder, assume y_test_res/y_train contain the original numeric labels.
                             # Determine the full set of classes from the TRAINING data.
                             try:
                                 # --- CORRECTED: Use y_train to find all classes the model was trained on ---
                                 if 'y_train' in st.session_state and st.session_state.y_train is not None:
                                     numeric_classes = np.unique(st.session_state.y_train) # Use y_train!
                                 else:
                                     # Fallback if y_train isn't available (shouldn't happen if trained)
                                     st.warning("y_train not found in session state. Falling back to y_test for class determination (may be incomplete).")
                                     numeric_classes = np.unique(y_test_res)

                                 # --- Safety Check: Ensure consistency with y_proba shape ---
                                 # This check is now more about verifying training consistency
                                 if len(numeric_classes) != n_classes:
                                     st.warning(f"Potential Class Inconsistency: Found {len(numeric_classes)} unique classes in y_train, but model predicts probabilities for {n_classes} classes. This might indicate issues during training or data processing. Inferring classes as 0 to {n_classes-1} for plotting.")
                                     numeric_classes = list(range(n_classes)) # Use range(n_classes) as the most direct inference from y_proba shape

                                 # Create generic string labels for plotting if original names aren't known
                                 all_classes = [f"Class {i}" for i in numeric_classes]

                             except Exception as e:
                                 st.error(f"Could not determine numeric classes for ROC: {e}")
                                 numeric_classes = None


                        if numeric_classes is not None:
                             # Binarize y_test using the definitive list of *numeric* classes.
                             try:
                                 y_test_binarized = label_binarize(y_test_res, classes=numeric_classes)

                                 # --- Verification Step ---
                                 if y_test_binarized.shape[1] != y_proba_res.shape[1]:
                                     st.error(f"CRITICAL SHAPE MISMATCH for ROC Curve:")
                                     st.error(f" - Binarized y_test has shape: {y_test_binarized.shape}")
                                     st.error(f" - Predicted probabilities (y_proba) has shape: {y_proba_res.shape}")
                                     st.error(f" - Determined numeric classes for binarization: {numeric_classes}")
                                     st.error("Cannot plot multi-class ROC. This usually indicates an issue with how classes were determined or inconsistency in the data.")
                                 else:
                                     # --- Proceed with plotting if shapes match ---
                                     fpr = dict()
                                     tpr = dict()
                                     roc_auc = dict()

                                     for i in range(n_classes): # Loop continues based on y_proba columns
                                         # Use the original class label if possible for the plot legend
                                         # Ensure all_classes has the same length as numeric_classes
                                         if all_classes is not None and len(all_classes) == len(numeric_classes):
                                            class_label = all_classes[i] # Get original label name
                                         else:
                                            class_label = f'Class {numeric_classes[i]}' # Fallback to numeric class

                                         # Calculate ROC using the correctly binarized labels
                                         fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba_res[:, i])
                                         try:
                                             # Calculate AUC, handle potential errors if fpr/tpr are degenerate
                                             roc_auc[i] = auc(fpr[i], tpr[i]) # Correct assignment
                                             ax_cls[1].plot(fpr[i], tpr[i], lw=2, label=f'ROC (Class {class_label}, AUC = {roc_auc[i]:.3f})') # Use roc_auc[i]
                                         except ValueError as auc_ve:
                                              st.warning(f"Could not calculate AUC for class {class_label}: {auc_ve}. May occur if class has only one sample or no variance in predictions.")
                                              # Plot ROC curve anyway, but indicate AUC is N/A
                                              ax_cls[1].plot(fpr[i], tpr[i], lw=2, label=f'ROC (Class {class_label}, AUC: N/A)')

                             except ValueError as bin_ve:
                                 # This error can happen if y_test_res contains values not in numeric_classes
                                 st.error(f"Error binarizing y_test for ROC curve: {bin_ve}.")
                                 st.error("This might happen if y_test contains labels unexpected by the model or the determined class list.")
                                 st.text(f"Labels found in y_test: {np.unique(y_test_res)}")
                                 st.text(f"Classes used for binarization: {numeric_classes}")
                             except Exception as roc_ex:
                                 # Catch other potential errors during ROC calculation
                                 st.error(f"An unexpected error occurred during multi-class ROC calculation: {roc_ex}")
                                 st.error(f"Traceback: {traceback.format_exc()}")

                        else:
                             st.warning("Could not reliably determine classes for multi-class ROC curve calculation.")
                else:
                    st.warning("ROC curve requires probability predictions (`predict_proba`), which were not available or failed for this model.")

                ax_cls[1].legend(loc="lower right")
                st.pyplot(fig_cls)
                plt.close(fig_cls) # Close figure

        with tab2:
             st.markdown("##### Feature Importance / Coefficients")
             if feature_importance_res is not None and not feature_importance_res.empty:
                 # Determine the correct column name ('Importance' or 'Coefficient Magnitude')
                 importance_col = feature_importance_res.columns[1] # Get the second column name

                 # Show top N features
                 max_features = len(feature_importance_res)
                 default_n = min(15, max_features) # Sensible default
                 n_features_to_show = st.slider("Number of top features to display:",
                                                 min_value=1,
                                                 max_value=max_features,
                                                 value=default_n,
                                                 key="n_features_slider")
                 top_features = feature_importance_res.head(n_features_to_show)

                 fig_imp, ax_imp = plt.subplots(figsize=(10, max(4, n_features_to_show * 0.4))) # Adjust height dynamically
                 sns.barplot(x=importance_col, y='Feature', data=top_features, ax=ax_imp, palette='viridis', orient='h')
                 ax_imp.set_title(f'Top {n_features_to_show} Features by {importance_col}')
                 ax_imp.set_xlabel(importance_col)
                 ax_imp.set_ylabel('Feature (Processed Name)')
                 plt.tight_layout() # Adjust layout
                 st.pyplot(fig_imp)
                 plt.close(fig_imp) # Close figure

                 with st.expander("Show all feature importances/coefficients"):
                     st.dataframe(feature_importance_res)
             else:
                 st.info("Feature importance or coefficients could not be calculated or are not applicable for this model.")

        with tab3:
             if model_type_res == "Classification":
                 st.markdown("##### Classification Report")
                 try:
                     # Get target names correctly
                     target_names_report = None
                     if target_encoder_res:
                         target_names_report = [str(cls) for cls in target_encoder_res.classes_]

                     report = classification_report(
                         y_test_res,
                         y_pred_res,
                         target_names=target_names_report,
                         output_dict=False, # Get string representation for display
                         zero_division=0
                     )
                     st.text(report)
                     # Provide explanation
                     st.caption("""
                     **Precision:** Of all instances predicted positive, what fraction was actually positive? (TP / (TP + FP))
                     **Recall (Sensitivity):** Of all actual positive instances, what fraction did the model correctly predict? (TP / (TP + FN))
                     **F1-Score:** Harmonic mean of Precision and Recall. Good balance metric. (2 * (Precision * Recall) / (Precision + Recall))
                     **Support:** Number of actual occurrences of the class in the test set.
                     **Accuracy:** Overall fraction of correct predictions. ((TP + TN) / Total)
                     **Macro Avg:** Average of metric for each class (unweighted).
                     **Weighted Avg:** Average of metric for each class, weighted by support.
                     """)

                 except Exception as e:
                     st.warning(f"Could not generate classification report: {e}")
             else: # Regression
                 st.markdown("##### Residuals Distribution")
                 residuals = y_test_res - y_pred_res
                 fig_res_dist, ax_res_dist = plt.subplots(figsize=(8, 5))
                 sns.histplot(residuals, kde=True, ax=ax_res_dist, bins=30)
                 ax_res_dist.set_title("Distribution of Residuals (Actual - Predicted)")
                 ax_res_dist.set_xlabel("Residual Value")
                 ax_res_dist.set_ylabel("Frequency")
                 ax_res_dist.axvline(np.mean(residuals), color='r', linestyle='--', label=f'Mean: {np.mean(residuals):.2f}') # Use np.mean
                 ax_res_dist.axvline(np.median(residuals), color='g', linestyle=':', label=f'Median: {np.median(residuals):.2f}') # Use np.median
                 ax_res_dist.legend()
                 st.pyplot(fig_res_dist)
                 plt.close(fig_res_dist) # Close figure
                 st.caption("Ideally, residuals should be normally distributed around zero, indicating that the model's errors are random.")


    # Add a message if training was attempted via submit but failed
    elif submitted and not st.session_state.model_trained:
         st.error("Model training was attempted but failed. Please check the error messages above and your configuration.")

# Add a footer or some other info if desired
st.divider()
st.caption("ML Model Trainer App - Built with Streamlit")