# ============================================================
# TASK 3 : Handwritten Character Recognition — Flask App
# CodeAlpha Internship
# ============================================================
# Lancer : python app.py  →  http://127.0.0.1:5000
# Dataset : sklearn digits (MNIST-style, 8x8, 1797 samples, 10 classes)
# ============================================================

from flask import Flask, render_template, jsonify, request
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

app  = Flask(__name__)
STATE = {}

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train():
    print("🖊️  Loading sklearn digits dataset …")
    digits = load_digits()
    X = digits.data / 16.0          # normalize to [0, 1]
    y = digits.target               # 0–9

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = {
        "MLP Neural Net": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu', solver='adam',
                max_iter=500, early_stopping=True,
                validation_fraction=0.1, random_state=42
            ))
        ]),
        "SVM (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=150, max_depth=None,
                random_state=42, n_jobs=1
            ))
        ]),
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=2000, random_state=42
            ))
        ]),
    }

    results = {}
    for name, pipe in pipelines.items():
        print(f"  Training {name} …")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')
        results[name] = {
            'pipeline': pipe,
            'y_pred':   y_pred.tolist(),
            'accuracy': round(acc, 4),
            'f1':       round(f1, 4),
            'cv_mean':  None,
            'cv_std':   None,
            'cv_all':   None,
        }

    best_name = max(results, key=lambda k: results[k]['accuracy'])

    # PCA 2D for scatter
    scaler_pca = StandardScaler()
    X_scaled   = scaler_pca.fit_transform(X)
    pca        = PCA(n_components=2, random_state=42)
    X_pca      = pca.fit_transform(X_scaled)
    idx        = np.random.choice(len(X), min(600, len(X)), replace=False)

    # Sample digit images (5 per class) for the gallery
    samples = {}
    for d in range(10):
        idxd = np.where(y == d)[0][:6]
        samples[str(d)] = digits.images[idxd].tolist()  # 8x8 pixel grids

    # Per-class distribution
    dist = {str(i): int(np.sum(y == i)) for i in range(10)}

    STATE.update({
        'digits':    digits,
        'X': X, 'y': y,
        'X_train':   X_train,
        'X_test':    X_test,
        'y_train':   y_train,
        'y_test':    y_test.tolist(),
        'results':   results,
        'best_name': best_name,
        'X_pca':     X_pca[idx],
        'y_pca':     y[idx].tolist(),
        'pca_var':   pca.explained_variance_ratio_.tolist(),
        'samples':   samples,
        'dist':      dist,
    })
    print(f"✅  Done — best: {best_name}  acc={results[best_name]['accuracy']}")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def api_models():
    return jsonify({
        'models':    list(STATE['results'].keys()),
        'best_name': STATE['best_name'],
    })

# ── Overview ─────────────────────────────────
@app.route('/api/overview')
def api_overview():
    results   = STATE['results']
    best_name = STATE['best_name']
    X, y      = STATE['X'], STATE['y']

    table = [{
        'name':     name,
        'accuracy': r['accuracy'],
        'f1':       r['f1'],
        'cv_mean':  r['cv_mean'] or '—',
        'cv_std':   r['cv_std']  or '—',
        'is_best':  name == best_name,
    } for name, r in results.items()]

    return jsonify({
        'n_samples':  int(len(y)),
        'n_features': int(X.shape[1]),
        'n_classes':  10,
        'image_size': '8×8',
        'best_name':  best_name,
        'best_acc':   results[best_name]['accuracy'],
        'best_f1':    results[best_name]['f1'],
        'dist':       STATE['dist'],
        'table':      table,
        'samples':    STATE['samples'],
    })

# ── Performance ───────────────────────────────
@app.route('/api/performance/<model_name>')
def api_performance(model_name):
    results = STATE['results']
    y_test  = STATE['y_test']

    if model_name not in results:
        return jsonify({'error': 'not found'}), 404

    res    = results[model_name]
    y_pred = res['y_pred']

    # Lazy CV
    if res['cv_mean'] is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            res['pipeline'], STATE['X_train'], STATE['y_train'],
            cv=cv, scoring='accuracy', n_jobs=1
        )
        res['cv_mean'] = round(float(cv_scores.mean()), 4)
        res['cv_std']  = round(float(cv_scores.std()),  4)
        res['cv_all']  = cv_scores.round(4).tolist()

    cm         = confusion_matrix(y_test, y_pred).tolist()
    cm_np      = np.array(cm)
    recall_pc  = (cm_np.diagonal() / cm_np.sum(axis=1)).round(4).tolist()
    f1_pc      = []
    for i in range(10):
        yt = (np.array(y_test) == i).astype(int)
        yp = (np.array(y_pred) == i).astype(int)
        from sklearn.metrics import f1_score as _f1
        f1_pc.append(round(float(_f1(yt, yp, zero_division=0)), 4))

    report = classification_report(
        y_test, y_pred,
        target_names=[str(i) for i in range(10)],
        output_dict=True
    )

    cv_data = {}
    for name, r in results.items():
        cv_data[name] = r['cv_all'] if r['cv_all'] else [r['accuracy']] * 3

    return jsonify({
        'model_name':       model_name,
        'accuracy':         res['accuracy'],
        'f1':               res['f1'],
        'cv_mean':          res['cv_mean'],
        'cv_std':           res['cv_std'],
        'confusion_matrix': cm,
        'recall_per_class': recall_pc,
        'f1_per_class':     f1_pc,
        'report':           report,
        'cv_boxplot':       cv_data,
    })

# ── PCA ───────────────────────────────────────
@app.route('/api/pca')
def api_pca():
    X_pca = STATE['X_pca']
    y_pca = STATE['y_pca']
    var   = STATE['pca_var']
    COLORS = ['#00f5d4','#f72585','#fee440','#00bbf9','#9b5de5',
              '#f15bb5','#fb5607','#3a86ff','#06d6a0','#ffbe0b']
    points = []
    for d in range(10):
        mask = np.array(y_pca) == d
        points.append({
            'digit': d,
            'color': COLORS[d],
            'x': X_pca[mask, 0].round(4).tolist(),
            'y': X_pca[mask, 1].round(4).tolist(),
        })
    return jsonify({
        'points':   points,
        'var_pc1':  round(var[0], 4),
        'var_pc2':  round(var[1], 4),
        'total_var':round(sum(var), 4),
    })

# ── Predict from canvas ───────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data       = request.get_json()
    model_name = data.get('model', STATE['best_name'])
    pixels     = data.get('pixels')   # list of 64 float values [0,1]

    if model_name not in STATE['results']:
        return jsonify({'error': 'not found'}), 404
    if not pixels or len(pixels) != 64:
        return jsonify({'error': 'need exactly 64 pixel values'}), 400

    feat     = np.array(pixels, dtype=float).reshape(1, -1)
    pipeline = STATE['results'][model_name]['pipeline']
    pred     = int(pipeline.predict(feat)[0])
    proba    = pipeline.predict_proba(feat)[0].round(4).tolist()

    COLORS = ['#00f5d4','#f72585','#fee440','#00bbf9','#9b5de5',
              '#f15bb5','#fb5607','#3a86ff','#06d6a0','#ffbe0b']

    return jsonify({
        'prediction':    pred,
        'probabilities': proba,
        'colors':        COLORS,
        'top3': sorted(
            [{'digit': i, 'prob': proba[i], 'color': COLORS[i]} for i in range(10)],
            key=lambda x: -x['prob']
        )[:3],
    })

# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀  Training models …")
    train()
    print("🌐  http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)