import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import joblib
import io
import traceback
from itertools import cycle

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Model Trainer App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Variables & Constants ---
DEFAULT_DATASETS = ["tips", "titanic", "iris", "diamonds", "penguins"]
CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest Classifier": RandomForestClassifier
}
REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Random Forest Regressor": RandomForestRegressor
}

# --- Helper Functions ---
@st.cache_data
def load_seaborn_dataset(dataset_name):
    try:
        if dataset_name == "titanic":
             df = sns.load_dataset(dataset_name); df['deck'] = df['deck'].astype(str).fillna('Unknown')
        else: df = sns.load_dataset(dataset_name)
        st.success(f"Loaded '{dataset_name}' dataset successfully!")
        return df.dropna(axis=1, how='all')
    except Exception as e: st.error(f"Error loading '{dataset_name}': {e}"); return None

@st.cache_data
def load_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded CSV loaded successfully!")
        return df.dropna(axis=1, how='all')
    except Exception as e: st.error(f"Error loading uploaded file: {e}"); return None

def identify_column_types(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    target = None; common = ['target','species','survived','tip','price','charges','quality','score','total_bill']
    for col in reversed(df.columns):
        if col.lower() in common: target = col; break
    if target is None and len(df.columns) > 0: target = df.columns[-1]
    if target not in df.columns: target = None
    return num, cat, target

def determine_task_type(df, target_column):
    if target_column is None or target_column not in df.columns: return None
    target = df[target_column].dropna();
    if target.empty: return None
    dtype = target.dtype; unique = target.nunique(); total = len(target)
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype): return "Classification"
    if pd.api.types.is_numeric_dtype(dtype):
        if pd.api.types.is_float_dtype(dtype) or (unique > 25 and (unique / total) > 0.05): return "Regression"
        else: return "Classification"
    return None

# --- Streamlit App UI ---
st.title("🧠 Interactive ML Model Trainer")
st.write("Select data, configure features & model, then train and evaluate.")

# Init session state
init_session_state = {'dataset_choice': None, 'trained_pipeline': None, 'current_dataset_id': None, 'data_source_choice': "Seaborn Dataset"}
for key, default_value in init_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- *** ADD INITIALIZATION FOR SUBMITTED *** ---
submitted = False # Initialize submitted state

# --- Sidebar ---
with st.sidebar:
    st.header("1. Data Selection")
    data_source_options = ["Seaborn Dataset", "Upload CSV"]
    try: current_choice_index = data_source_options.index(st.session_state.data_source_choice)
    except ValueError: current_choice_index = 0

    data_source = st.radio("Choose data source:", options=data_source_options, index=current_choice_index, key="data_source_radio")
    if st.session_state.data_source_choice != data_source:
        st.session_state.data_source_choice = data_source
        st.session_state.trained_pipeline = None; st.session_state.current_dataset_id = None
        st.rerun()

    df = None; dataset_name_or_upload = None
    if data_source == "Seaborn Dataset":
        selected_seaborn_dataset = st.selectbox("Select Seaborn Dataset", DEFAULT_DATASETS, index=None, placeholder="Choose a dataset...", key="dataset_select")
        if selected_seaborn_dataset:
            dataset_name_or_upload = selected_seaborn_dataset
            if st.session_state.current_dataset_id != selected_seaborn_dataset:
                 st.session_state.trained_pipeline = None; st.session_state.current_dataset_id = selected_seaborn_dataset
            df = load_seaborn_dataset(selected_seaborn_dataset)
            st.session_state.dataset_choice = selected_seaborn_dataset
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="file_uploader")
        if uploaded_file is not None:
            dataset_name_or_upload = uploaded_file.name
            if st.session_state.current_dataset_id != uploaded_file.name:
                 st.session_state.trained_pipeline = None; st.session_state.current_dataset_id = uploaded_file.name
            df = load_uploaded_file(uploaded_file)
            st.session_state.dataset_choice = "Uploaded CSV"

    # --- Feature Eng & Model Config (only if data is loaded) ---
    if df is not None:
        st.sidebar.header("2. Feature Engineering")
        numerical_cols, categorical_cols, potential_target = identify_column_types(df)
        available_targets = ["<Select>"] + df.columns.tolist()
        target_index = 0
        if potential_target and potential_target in available_targets:
             try: target_index = available_targets.index(potential_target)
             except ValueError: target_index = 0
        target_variable = st.sidebar.selectbox("Select Target Variable (Y)", available_targets, index=target_index, key="target_select")

        task_type = None
        if target_variable and target_variable != "<Select>":
             task_type = determine_task_type(df, target_variable)
             if task_type: st.sidebar.info(f"Task Type: **{task_type}**")
             else: st.sidebar.warning("Could not determine task type.")
        else: st.sidebar.info("Select target variable.")

        selected_features = []
        if target_variable and target_variable != "<Select>":
            available_features = [col for col in df.columns if col != target_variable]
            num_opts = sorted([col for col in numerical_cols if col in available_features])
            cat_opts = sorted([col for col in categorical_cols if col in available_features])
            st.sidebar.markdown("**Select Predictor Variables (X)**")
            sel_num = st.sidebar.multiselect("Numerical Features", num_opts, default=num_opts, key="num_features_select")
            sel_cat = st.sidebar.multiselect("Categorical Features", cat_opts, default=cat_opts, key="cat_features_select")
            selected_features = sorted(sel_num + sel_cat)
        else: st.sidebar.markdown("**Select Predictor Variables (X)**"); st.sidebar.write("(Select target first)")

        ready_for_config = (df is not None and target_variable and target_variable != "<Select>" and selected_features and task_type is not None)
        if not ready_for_config and dataset_name_or_upload:
             if not target_variable or target_variable == "<Select>": st.sidebar.warning("Select target.")
             elif not selected_features: st.sidebar.warning("Select features.")
             elif task_type is None: st.sidebar.warning("Task type undetermined.")

        with st.sidebar.form("training_form"):
            st.header("3. Model Configuration")
            model_choice = None; available_models = {}
            if task_type == "Regression":
                available_models = REGRESSION_MODELS
                model_choice = st.selectbox("Select Regression Model", list(available_models.keys()), key="model_select", disabled=not ready_for_config)
            elif task_type == "Classification":
                available_models = CLASSIFICATION_MODELS
                model_choice = st.selectbox("Select Classification Model", list(available_models.keys()), key="model_select", disabled=not ready_for_config)
            else: st.markdown("*Select target/features first*")

            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="test_size_slider", disabled=not ready_for_config)
            random_state = st.number_input("Random State", value=42, step=1, key="random_state_input", disabled=not ready_for_config)

            hyperparameters = {}
            if model_choice:
                 st.markdown(f"**Hyperparameters for {model_choice}**")
                 if "Random Forest" in model_choice:
                      hyperparameters['n_estimators'] = st.slider("Estimators", 50, 500, 100, 10, key="rf_n_estimators", disabled=not ready_for_config)
                      max_depth_option = st.slider("Max Depth (31=None)", 3, 31, 10, 1, key="rf_max_depth", disabled=not ready_for_config)
                      hyperparameters['max_depth'] = None if max_depth_option == 31 else max_depth_option
                      hyperparameters['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2, 1, key="rf_min_samples_split", disabled=not ready_for_config)
                      hyperparameters['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 20, 1, 1, key="rf_min_samples_leaf", disabled=not ready_for_config)
                 elif model_choice == "Logistic Regression":
                       hyperparameters['max_iter'] = st.number_input("Max Iterations", 100, 1000, 100, 100, key="lr_max_iter", disabled=not ready_for_config)

            # submitted gets potentially reassigned to True here when button is clicked
            submitted = st.form_submit_button("Fit Model", disabled=not ready_for_config or not model_choice)

# --- Main Area Display Logic ---
if df is not None:
    st.header("📊 Data Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")
    with st.expander("Data Info"):
            buffer = io.StringIO(); df.info(buf=buffer); st.text(buffer.getvalue())
else:
     st.info("👈 Choose a data source and dataset/file in the sidebar to begin.")

# --- Model Training (triggered by form submission) ---
# Now 'submitted' is guaranteed to be defined (either False or True from the form)
if submitted and df is not None: # Ensure df exists before proceeding
    st.header("⏳ Training In Progress...")
    progress_bar = st.progress(0, text="Initializing...")
    st.session_state['trained_pipeline'] = None

    try:
        progress_bar.progress(5, text="Preparing data...")
        X = df[selected_features].copy()
        y = df[target_variable].copy()

        rows_dropped = False
        if y.isnull().any():
            na_indices = y.isnull()
            st.warning(f"Target '{target_variable}' has {na_indices.sum()} missing values. Rows dropped.")
            X = X.loc[~na_indices]; y = y.loc[~na_indices]
            rows_dropped = True
        X.reset_index(drop=True, inplace=True); y.reset_index(drop=True, inplace=True)
        if rows_dropped: st.info("Indices reset after dropping NaN targets.")
        if not X.index.equals(y.index): raise ValueError("Index mismatch post-reset.")

        label_encoder = None; st.session_state['label_encoder_classes'] = None; num_classes = 0
        if task_type == "Classification":
             if not pd.api.types.is_numeric_dtype(y.dtype) or y.min() != 0:
                 le = LabelEncoder(); y = le.fit_transform(y)
                 st.info(f"Target encoded. Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                 label_encoder = le; st.session_state['label_encoder_classes'] = list(le.classes_)
             num_classes = len(np.unique(y))

        progress_bar.progress(10, text="Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if task_type=="Classification" and num_classes > 1 else None)

        progress_bar.progress(20, text="Setting up preprocessing...")
        num_feat_train = X_train.select_dtypes(include=np.number).columns.tolist()
        cat_feat_train = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        num_proc = sorted([f for f in num_feat_train if f in selected_features])
        cat_proc = sorted([f for f in cat_feat_train if f in selected_features])
        num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        transformers = []
        if num_proc: transformers.append(('num', num_pipe, num_proc))
        if cat_proc: transformers.append(('cat', cat_pipe, cat_proc))
        if not transformers: raise ValueError("No features to process.")
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough', verbose_feature_names_out=False)
        try: preprocessor.set_output(transform="pandas")
        except AttributeError: pass

        progress_bar.progress(30, text="Defining model...")
        model_class = available_models[model_choice]
        model_params = hyperparameters.copy()
        sig = model_class.__init__.__code__.co_varnames
        if "random_state" in sig: model_params['random_state'] = random_state
        model = model_class(**model_params)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        progress_bar.progress(50, text="Fitting model...")
        pipeline.fit(X_train, y_train)
        progress_bar.progress(80, text="Making predictions...")

        y_pred = pipeline.predict(X_test)
        y_pred_proba_all, y_pred_proba_binary = None, None
        st.session_state['y_pred_proba_all'] = None
        if task_type == "Classification" and hasattr(pipeline, "predict_proba"):
            try:
                pred_proba_all = pipeline.predict_proba(X_test)
                st.session_state['y_pred_proba_all'] = pred_proba_all
                if pred_proba_all.shape[1] == 2:
                    pipeline_classes = getattr(pipeline, 'classes_', None)
                    pos_idx = np.where(pipeline_classes == 1)[0] if pipeline_classes is not None else [1]
                    idx = pos_idx[0] if len(pos_idx) > 0 else 1
                    y_pred_proba_binary = pred_proba_all[:, idx]
                    st.session_state['y_pred_proba'] = y_pred_proba_binary
            except Exception as e: st.warning(f"Could not get probabilities: {e}")
        elif task_type == "Classification": st.info(f"{model_choice} doesn't support predict_proba.")

        progress_bar.progress(90, text="Evaluating results...")

        feature_names_out = []
        try:
            prep_step = pipeline.named_steps['preprocessor']
            if hasattr(prep_step, 'get_feature_names_out'): feature_names_out = list(prep_step.get_feature_names_out())
            else: raise AttributeError("get_feature_names_out not available.")
        except Exception as e_feat:
             st.warning(f"Could not get feature names reliably ({e_feat}). Using fallback.")
             try:
                 num_out = pipeline.named_steps['preprocessor'].transform(X_test).shape[1]
                 feature_names_out = [f"Feature_{i}" for i in range(num_out)]
             except Exception as e_fb:
                  st.error(f"Fallback failed: {e_fb}"); feature_names_out = list(selected_features)
        if not feature_names_out and (hasattr(pipeline.named_steps['model'], 'feature_importances_') or hasattr(pipeline.named_steps['model'], 'coef_')):
            st.error("Critical: No feature names for importance plot.")
            model_step=pipeline.named_steps['model']
            imp_len = len(getattr(model_step,'feature_importances_',[])) or getattr(model_step,'coef_',np.array([])).shape[-1]
            feature_names_out = [f"Unknown_{i}" for i in range(imp_len)]

        st.session_state.update({
            'trained_pipeline': pipeline, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred,
            'model_choice': model_choice, 'task_type': task_type, 'selected_features': selected_features,
            'feature_names_out': list(feature_names_out), 'label_encoder': label_encoder, 'num_classes': num_classes
        })
        st.success("✅ Model Training Completed Successfully!")
        progress_bar.progress(100, text="Done!")

    except Exception as e:
        st.error(f"An error occurred during training: {e}")
        st.error(traceback.format_exc())
        st.session_state['trained_pipeline'] = None
        for key in ['metrics', 'feature_importances']:
             if key in st.session_state: del st.session_state[key]
    finally: progress_bar.empty()

elif submitted and df is None:
     st.error("Cannot train model: No data loaded.")

# --- Display Results ---
if st.session_state.get('trained_pipeline') is not None:
    st.header("📈 Model Evaluation & Results")
    pipeline = st.session_state.trained_pipeline; y_test = st.session_state.y_test; y_pred = st.session_state.y_pred
    y_pred_proba_all = st.session_state.get('y_pred_proba_all'); task_type = st.session_state.task_type
    feature_names_out = st.session_state.feature_names_out; label_encoder = st.session_state.label_encoder
    num_classes = st.session_state.num_classes; model_choice = st.session_state.model_choice

    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    if task_type == "Regression":
        try:
            mse=mean_squared_error(y_test, y_pred); rmse=np.sqrt(mse); r2=r2_score(y_test, y_pred)
            with col1: st.metric("MSE", f"{mse:.4f}"); st.metric("RMSE", f"{rmse:.4f}")
            with col2: st.metric("R²", f"{r2:.4f}")
            st.session_state['metrics'] = {"MSE": mse, "RMSE": rmse, "R2": r2}
        except Exception as e: st.error(f"Error calc regression metrics: {e}")
    elif task_type == "Classification":
        try:
            accuracy=accuracy_score(y_test, y_pred)
            names = [str(cls) for cls in label_encoder.classes_] if label_encoder and hasattr(label_encoder, 'classes_') else None
            report=classification_report(y_test, y_pred, output_dict=True, target_names=names, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            with col1: st.metric("Accuracy", f"{accuracy:.4f}")
            st.text("Classification Report:"); st.dataframe(report_df)
            st.session_state['metrics'] = {"Accuracy": accuracy, "Classification Report": report}
        except Exception as e: st.error(f"Error calc classification metrics: {e}")

    st.subheader("Visualizations")
    plot_fns = []
    if task_type == "Regression": plot_fns.append("plot_residuals")
    if task_type == "Classification":
        plot_fns.append("plot_confusion_matrix")
        if y_pred_proba_all is not None: plot_fns.append("plot_roc_curve")
        else: st.info("ROC Curve requires probabilities (unavailable).")
    model_step = pipeline.named_steps['model']
    has_imp = hasattr(model_step, 'feature_importances_') or hasattr(model_step, 'coef_')
    has_names = feature_names_out and not all("Unknown_" in fn for fn in feature_names_out)
    if has_imp and has_names: plot_fns.append("plot_feature_importance")
    elif has_imp and not has_names: st.warning("Cannot plot importance: feature names unknown.")

    n_plots = len(plot_fns)
    if n_plots > 0:
        n_cols=2; n_rows=(n_plots+n_cols-1)//n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows), squeeze=False)
        axs = axs.flatten(); plot_idx = 0

        def plot_residuals(ax):
            try: sns.histplot(y_test - y_pred, kde=True, ax=ax); ax.set_title('Residuals Dist'); ax.set_xlabel('Residual'); ax.set_ylabel('Freq')
            except Exception as e: ax.text(0.5,0.5,f'Err Res:\n{e}',ha='center',va='center'); ax.axis('off')
        def plot_confusion_matrix(ax):
            try:
                cm=confusion_matrix(y_test, y_pred); lbls=[str(c) for c in label_encoder.classes_] if label_encoder else 'auto'
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=lbls, yticklabels=lbls)
                ax.set_title('Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            except Exception as e: ax.text(0.5,0.5,f'Err CM:\n{e}',ha='center',va='center'); ax.axis('off')
        def plot_roc_curve(ax):
            try:
                y_t, y_p, n_cls, pipe, le = st.session_state.y_test, st.session_state.y_pred_proba_all, st.session_state.num_classes, st.session_state.trained_pipeline, st.session_state.label_encoder
                p_cls = getattr(pipe, 'classes_', None)
                labels = [str(c) for c in le.classes_] if le and hasattr(le,'classes_') else ([str(c) for c in p_cls] if p_cls is not None else [f'Cls {i}' for i in range(n_cls)])
                if len(labels) != n_cls: labels = [f'Cls {i}' for i in range(n_cls)]
                if n_cls == 2:
                    y_p_bin = st.session_state.get('y_pred_proba')
                    if y_p_bin is None: idx = np.where(p_cls == 1)[0][0] if p_cls is not None and 1 in p_cls else 1; y_p_bin = y_p[:, idx]
                    fpr, tpr, _ = roc_curve(y_t, y_p_bin); roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})'); ax.plot([0,1],[0,1],'navy',lw=2,ls='--'); ax.set_title('ROC Curve'); ax.legend(loc="lower right")
                elif n_cls > 2:
                    y_t_bin = label_binarize(y_t, classes=p_cls if p_cls is not None else np.arange(n_cls))
                    if y_t_bin.shape[1] != n_cls: y_t_bin = label_binarize(y_t, classes=np.arange(n_cls))
                    fpr, tpr, roc_auc = {}, {}, {}
                    for i in range(n_cls): fpr[i], tpr[i], _ = roc_curve(y_t_bin[:,i], y_p[:,i]); roc_auc[i] = auc(fpr[i], tpr[i])
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_t_bin.ravel(), y_p.ravel()); roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    ax.plot(fpr["micro"], tpr["micro"], label=f'Micro-avg (AUC={roc_auc["micro"]:.2f})', color='deeppink', ls=':', lw=4)
                    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
                    for i, color in zip(range(n_cls), colors): ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{labels[i]} (AUC={roc_auc[i]:.2f})')
                    ax.plot([0,1],[0,1],'k--',lw=2); ax.set_title('Multiclass ROC (OvR)'); ax.legend(loc="lower right", fontsize='small')
                else: ax.text(0.5,0.5,'ROC needs >= 2 classes.',ha='center',va='center'); ax.axis('off')
                ax.set_xlim([0.0,1.0]); ax.set_ylim([0.0,1.05]); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
            except Exception as e: ax.text(0.5,0.5,f'Err ROC:\n{e}',ha='center',va='center',wrap=True); ax.axis('off'); st.error(f"ROC Err: {e}\n{traceback.format_exc()}")
        def plot_feature_importance(ax):
            try:
                imp = None; imp_t = ""
                step = pipeline.named_steps['model']
                if hasattr(step, 'feature_importances_'): imp = step.feature_importances_; imp_t = "Importance"
                elif hasattr(step, 'coef_'): c = step.coef_; imp = np.mean(np.abs(c), axis=0) if c.ndim > 1 else np.abs(c); imp_t = "Coef (Abs)"
                if imp is not None:
                    if len(feature_names_out) == len(imp):
                        df_imp=pd.DataFrame({'Feature':feature_names_out,'Value':imp}).sort_values('Value',ascending=False).head(15)
                        sns.barplot(x='Value', y='Feature', data=df_imp, ax=ax, palette="viridis")
                        ax.set_title(f'Top {len(df_imp)} Feature {imp_t}s'); ax.set_xlabel(imp_t); ax.set_ylabel('Feature')
                        st.session_state['feature_importances'] = df_imp
                    else: msg = f'Plot Err: Names ({len(feature_names_out)}) != Values ({len(imp)})'; ax.text(0.5,0.5,msg,ha='center',va='center'); ax.axis('off'); st.error(msg)
                else: ax.text(0.5,0.5,'Cannot extract importance.',ha='center',va='center'); ax.axis('off')
            except Exception as e: ax.text(0.5,0.5,f'Err Imp:\n{e}',ha='center',va='center'); ax.axis('off')

        for plot_name in plot_fns:
            if plot_idx < len(axs):
                 func = locals().get(plot_name)
                 if func: func(axs[plot_idx])
                 plot_idx += 1
            else: st.warning("Exceeded subplot axes.")
        for i in range(plot_idx, len(axs)): axs[i].axis('off')
        plt.tight_layout(); st.pyplot(fig)
    else: st.info("No visualizations generated.")

    st.header("💾 Export Trained Model")
    st.write("Download model pipeline (`.joblib`).")
    try:
        export_obj = {'pipeline': pipeline, 'label_encoder': label_encoder}
        buffer = io.BytesIO(); joblib.dump(export_obj, buffer); buffer.seek(0)
        ds_id = st.session_state.get('dataset_choice', 'custom')
        if isinstance(ds_id, str) and ds_id.lower().endswith('.csv'): ds_id = ds_id[:-4]
        st.download_button(label="Download Model Bundle (.joblib)", data=buffer,
                           file_name=f"{ds_id}_{model_choice.replace(' ','_').lower()}_bundle.joblib",
                           mime="application/octet-stream", key="download_btn")
    except Exception as e: st.error(f"Error preparing download: {e}")

# --- Footer ---
st.markdown("---"); st.markdown("ML Model Trainer App")