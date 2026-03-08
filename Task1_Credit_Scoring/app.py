from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# ─────────────────────────────────────────────
# GLOBAL STATE — trained once at startup
# ─────────────────────────────────────────────
MODEL_STATE = {}

def train_models():
    np.random.seed(42)
    N = 5000

    age                = np.random.randint(21, 70, N)
    income             = np.random.normal(45000, 20000, N).clip(10000, 200000).astype(int)
    employment_years   = np.random.randint(0, 40, N)
    debt_amount        = np.random.normal(15000, 12000, N).clip(0, 100000).astype(int)
    num_credit_lines   = np.random.randint(1, 10, N)
    num_late_payments  = np.random.randint(0, 15, N)
    credit_utilization = np.random.uniform(0.0, 1.0, N).round(2)
    savings_balance    = np.random.normal(10000, 8000, N).clip(0, 80000).astype(int)
    loan_amount        = np.random.normal(20000, 15000, N).clip(1000, 150000).astype(int)
    loan_duration      = np.random.choice([12, 24, 36, 48, 60, 72], N)
    housing_status     = np.random.choice(['own', 'rent', 'mortgage'], N, p=[0.3, 0.35, 0.35])
    education          = np.random.choice(['high_school', 'bachelor', 'master', 'phd'], N, p=[0.3, 0.4, 0.2, 0.1])

    credit_score = (
        (income / 1000) * 0.3 - (debt_amount / 1000) * 0.25
        - num_late_payments * 3 - credit_utilization * 20
        + employment_years * 1.5 + (savings_balance / 1000) * 0.5
        + num_credit_lines * 1.0
        + np.where(housing_status == 'own', 10, np.where(housing_status == 'mortgage', 5, 0))
        + np.where(education == 'phd', 8, np.where(education == 'master', 5,
           np.where(education == 'bachelor', 2, 0)))
        + np.random.normal(0, 5, N)
    )
    threshold    = np.percentile(credit_score, 35)
    creditworthy = (credit_score > threshold).astype(int)

    df = pd.DataFrame({
        'age': age, 'income': income, 'employment_years': employment_years,
        'debt_amount': debt_amount, 'num_credit_lines': num_credit_lines,
        'num_late_payments': num_late_payments, 'credit_utilization': credit_utilization,
        'savings_balance': savings_balance, 'loan_amount': loan_amount,
        'loan_duration_months': loan_duration, 'housing_status': housing_status,
        'education': education, 'creditworthy': creditworthy
    })

    df['debt_to_income_ratio']        = (df['debt_amount'] / df['income']).round(4)
    df['loan_to_income_ratio']        = (df['loan_amount'] / df['income']).round(4)
    df['monthly_repayment']           = (df['loan_amount'] / df['loan_duration_months']).round(2)
    df['repayment_to_monthly_income'] = (df['monthly_repayment'] / (df['income'] / 12)).round(4)
    df['financial_stability_score']   = (
        df['employment_years'] * 2 + (df['savings_balance'] / 1000)
        - df['num_late_payments'] * 5 - df['credit_utilization'] * 10
    ).round(2)

    df = pd.get_dummies(df, columns=['housing_status', 'education'], drop_first=False)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop('creditworthy', axis=1)
    y = df['creditworthy']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = {
        "Logistic Regression": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', DecisionTreeClassifier(max_depth=8, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
        ])
    }

    results = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            'model':     pipeline,
            'y_pred':    y_pred.tolist(),
            'y_proba':   y_proba.tolist(),
            'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall':    round(recall_score(y_test, y_pred), 4),
            'F1-Score':  round(f1_score(y_test, y_pred), 4),
            'ROC-AUC':   round(roc_auc_score(y_test, y_proba), 4),
        }

    best_name = max(results, key=lambda k: results[k]['ROC-AUC'])

    MODEL_STATE['df']        = df
    MODEL_STATE['X']         = X
    MODEL_STATE['y']         = y
    MODEL_STATE['X_train']   = X_train
    MODEL_STATE['X_test']    = X_test
    MODEL_STATE['y_train']   = y_train
    MODEL_STATE['y_test']    = y_test.tolist()
    MODEL_STATE['results']   = results
    MODEL_STATE['best_name'] = best_name
    print(f"✅  Models trained — best: {best_name}")

# ─────────────────────────────────────────────
# ROUTES — HTML
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ─────────────────────────────────────────────
# API — OVERVIEW
# ─────────────────────────────────────────────
@app.route('/api/overview')
def api_overview():
    df      = MODEL_STATE['df']
    X       = MODEL_STATE['X']
    y       = MODEL_STATE['y']
    results = MODEL_STATE['results']
    best    = MODEL_STATE['best_name']

    models_table = []
    for name, res in results.items():
        models_table.append({
            'name':      name,
            'Accuracy':  res['Accuracy'],
            'Precision': res['Precision'],
            'Recall':    res['Recall'],
            'F1-Score':  res['F1-Score'],
            'ROC-AUC':   res['ROC-AUC'],
            'is_best':   name == best,
        })

    sample_cols = ['age','income','employment_years','debt_amount',
                   'num_late_payments','credit_utilization','savings_balance',
                   'debt_to_income_ratio','creditworthy']
    sample_data = df[sample_cols].head(8).to_dict(orient='records')

    best_res = results[best]
    return jsonify({
        'total_samples': len(df),
        'n_features':    X.shape[1],
        'good_credit':   int(y.sum()),
        'bad_credit':    int((y == 0).sum()),
        'best_name':     best,
        'best_metrics': {
            'Accuracy':  best_res['Accuracy'],
            'Precision': best_res['Precision'],
            'Recall':    best_res['Recall'],
            'F1-Score':  best_res['F1-Score'],
            'ROC-AUC':   best_res['ROC-AUC'],
        },
        'models_table': models_table,
        'sample_data':  sample_data,
    })

# ─────────────────────────────────────────────
# API — DATA ANALYSIS
# ─────────────────────────────────────────────
@app.route('/api/analysis')
def api_analysis():
    df  = MODEL_STATE['df']

    def hist_data(col, n_bins=30):
        good = df[df['creditworthy'] == 1][col]
        bad  = df[df['creditworthy'] == 0][col]
        all_vals = pd.concat([good, bad])
        bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
        def to_hist(series):
            counts, edges = np.histogram(series, bins=bins, density=True)
            centers = ((edges[:-1] + edges[1:]) / 2).round(4).tolist()
            return centers, counts.round(4).tolist()
        g_x, g_y = to_hist(good)
        b_x, b_y = to_hist(bad)
        return {'labels': g_x, 'good': g_y, 'bad': b_y}

    features = ['income', 'debt_amount', 'savings_balance',
                'credit_utilization', 'num_late_payments', 'employment_years']
    distributions = {f: hist_data(f) for f in features}

    # Correlation matrix
    num_cols = ['income','debt_amount','credit_utilization','savings_balance',
                'num_late_payments','debt_to_income_ratio',
                'financial_stability_score','creditworthy']
    corr = df[num_cols].corr().round(3)
    correlation = {
        'labels': num_cols,
        'matrix': corr.values.tolist()
    }

    # Feature engineering charts
    fe_dti   = hist_data('debt_to_income_ratio')
    fe_fin   = hist_data('financial_stability_score')
    fe_lti   = hist_data('loan_to_income_ratio')

    return jsonify({
        'distributions': distributions,
        'correlation':   correlation,
        'fe_dti':        fe_dti,
        'fe_fin':        fe_fin,
        'fe_lti':        fe_lti,
    })

# ─────────────────────────────────────────────
# API — MODEL PERFORMANCE
# ─────────────────────────────────────────────
@app.route('/api/performance/<model_name>')
def api_performance(model_name):
    results = MODEL_STATE['results']
    y_test  = MODEL_STATE['y_test']
    X       = MODEL_STATE['X']

    if model_name not in results:
        return jsonify({'error': 'Model not found'}), 404

    res    = results[model_name]
    y_pred = res['y_pred']

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC curves for ALL models
    roc_curves = {}
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        step = max(1, len(fpr) // 100)
        roc_curves[name] = {
            'fpr': fpr[::step].tolist(),
            'tpr': tpr[::step].tolist(),
            'auc': r['ROC-AUC'],
        }

    # Feature importance
    feature_importance = None
    clf = res['model'].named_steps.get('clf')
    if hasattr(clf, 'feature_importances_'):
        imps = clf.feature_importances_
        series = pd.Series(imps, index=X.columns).sort_values(ascending=False).head(15)
        feature_importance = {
            'features': series.index.tolist(),
            'values':   series.round(4).tolist(),
        }

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=['Bad Credit', 'Good Credit'],
        output_dict=True
    )

    return jsonify({
        'model_name':         model_name,
        'metrics': {
            'Accuracy':  res['Accuracy'],
            'Precision': res['Precision'],
            'Recall':    res['Recall'],
            'F1-Score':  res['F1-Score'],
            'ROC-AUC':   res['ROC-AUC'],
        },
        'confusion_matrix':   cm,
        'roc_curves':         roc_curves,
        'feature_importance': feature_importance,
        'report':             report,
    })

# ─────────────────────────────────────────────
# API — INDIVIDUAL PREDICTION
# ─────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data    = request.get_json()
    results = MODEL_STATE['results']
    X       = MODEL_STATE['X']

    model_name = data.get('model', MODEL_STATE['best_name'])
    if model_name not in results:
        return jsonify({'error': 'Model not found'}), 404

    age               = int(data['age'])
    income            = int(data['income'])
    employment_years  = int(data['employment_years'])
    debt_amount       = int(data['debt_amount'])
    num_credit_lines  = int(data['num_credit_lines'])
    num_late_payments = int(data['num_late_payments'])
    credit_utilization = float(data['credit_utilization'])
    savings_balance   = int(data['savings_balance'])
    loan_amount       = int(data['loan_amount'])
    loan_duration     = int(data['loan_duration'])
    housing_status    = data['housing_status']
    education         = data['education']

    dti         = debt_amount / income
    lti         = loan_amount / income
    monthly_rep = loan_amount / loan_duration
    rep_ratio   = monthly_rep / (income / 12)
    fin_score   = (employment_years * 2 + (savings_balance / 1000)
                   - num_late_payments * 5 - credit_utilization * 10)

    client = {
        'age': age, 'income': income, 'employment_years': employment_years,
        'debt_amount': debt_amount, 'num_credit_lines': num_credit_lines,
        'num_late_payments': num_late_payments, 'credit_utilization': credit_utilization,
        'savings_balance': savings_balance, 'loan_amount': loan_amount,
        'loan_duration_months': loan_duration,
        'debt_to_income_ratio': dti, 'loan_to_income_ratio': lti,
        'monthly_repayment': monthly_rep, 'repayment_to_monthly_income': rep_ratio,
        'financial_stability_score': fin_score,
        'housing_status_mortgage': 1 if housing_status == 'mortgage' else 0,
        'housing_status_own':      1 if housing_status == 'own'      else 0,
        'housing_status_rent':     1 if housing_status == 'rent'     else 0,
        'education_bachelor':      1 if education == 'bachelor'      else 0,
        'education_high_school':   1 if education == 'high_school'   else 0,
        'education_master':        1 if education == 'master'        else 0,
        'education_phd':           1 if education == 'phd'           else 0,
    }

    client_df = pd.DataFrame([client])
    for col in X.columns:
        if col not in client_df.columns:
            client_df[col] = 0
    client_df = client_df[X.columns]

    pipeline   = results[model_name]['model']
    prediction = int(pipeline.predict(client_df)[0])
    proba      = pipeline.predict_proba(client_df)[0]

    return jsonify({
        'prediction':  prediction,
        'prob_good':   round(float(proba[1]), 4),
        'prob_bad':    round(float(proba[0]), 4),
        'fin_score':   round(fin_score, 2),
        'dti':         round(dti, 4),
        'lti':         round(lti, 4),
        'monthly_rep': round(monthly_rep, 2),
        'rep_ratio':   round(rep_ratio, 4),
    })

# ─────────────────────────────────────────────
# API — MODEL NAMES
# ─────────────────────────────────────────────
@app.route('/api/models')
def api_models():
    return jsonify({
        'models':    list(MODEL_STATE['results'].keys()),
        'best_name': MODEL_STATE['best_name'],
    })

# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀  Training models…")
    train_models()
    print("🌐  Starting Flask at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)