# ============================================================
# TASK 2 : Emotion Recognition from Speech — Flask App
# CodeAlpha Internship
# ============================================================
# Lancer : python app.py  →  http://127.0.0.1:5000
# ============================================================

from flask import Flask, render_template, jsonify, request
import numpy as np
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
np.random.seed(42)

EMOTIONS = {
    '01': 'neutral',  '02': 'calm',     '03': 'happy',    '04': 'sad',
    '05': 'angry',    '06': 'fearful',  '07': 'disgust',  '08': 'surprised'
}
EMOTION_NAMES = list(EMOTIONS.values())
N_CLASSES = 8
N_MFCC    = 40

EMO_COLORS = ['#6366F1','#10B981','#F59E0B','#EF4444',
               '#EC4899','#14B8A6','#8B5CF6','#F97316']

# ─────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────
def simulate_features(emotion_id, n_mfcc=N_MFCC, noise=0.12):
    profiles = {
        0: (0.00, 0.40, 0.07), 1: (0.05, 0.35, 0.06),
        2: (0.70, 0.80, 0.22), 3: (-0.60, 0.25, 0.10),
        4: (0.90, 0.95, 0.32), 5: (-0.15, 0.55, 0.28),
        6: (-0.25, 0.45, 0.18), 7: (0.45, 0.75, 0.24),
    }
    bias, pitch, var = profiles[emotion_id]
    mfcc_mean = bias + np.array([pitch * np.sin(i / n_mfcc * np.pi) for i in range(n_mfcc)])
    mfcc_std  = var  + np.abs(np.random.normal(0, noise, n_mfcc))
    d1_mean   = np.gradient(mfcc_mean) + np.random.normal(0, noise, n_mfcc)
    d1_std    = mfcc_std * 0.6 + np.abs(np.random.normal(0, noise / 2, n_mfcc))
    d2_mean   = np.gradient(d1_mean)   + np.random.normal(0, noise, n_mfcc)
    d2_std    = mfcc_std * 0.4 + np.abs(np.random.normal(0, noise / 2, n_mfcc))
    chroma_m  = np.random.normal(pitch, var, 12)
    chroma_s  = np.abs(np.random.normal(var, noise, 12))
    return np.concatenate([
        mfcc_mean + np.random.normal(0, noise, n_mfcc), mfcc_std,
        d1_mean, d1_std, d2_mean, d2_std, chroma_m, chroma_s,
        [abs(bias) * 0.5 + np.random.normal(0, noise)],
        [abs(var)  + np.random.normal(0, noise / 2)],
        [abs(bias) * 0.3 + np.random.normal(0.1, noise)],
        [abs(var)  * 0.5 + np.random.normal(0, noise / 2)]
    ])

def generate_dataset(n_per_class=250):
    X, y = [], []
    for emo_id in range(N_CLASSES):
        for _ in range(n_per_class):
            X.append(simulate_features(emo_id))
            y.append(emo_id)
    X, y = np.array(X), np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
STATE = {}

def train():
    print("🔊  Generating MFCC features …")
    X, y = generate_dataset(n_per_class=250)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = {
        "SVM (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=42))
        ]),
        "MLP Neural Net": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu', solver='adam',
                learning_rate_init=0.001, max_iter=300,
                early_stopping=True, validation_fraction=0.1, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=12,
                min_samples_split=4, random_state=42, n_jobs=1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=80, learning_rate=0.15,
                max_depth=4, subsample=0.8, random_state=42
            ))
        ]),
    }

    results = {}

    for name, pipe in pipelines.items():
        print(f"  Training {name} …")
        pipe.fit(X_train, y_train)
        y_pred    = pipe.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average='weighted')
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

    # PCA for visualisation (sample 800 pts max)
    scaler_pca = StandardScaler()
    X_scaled   = scaler_pca.fit_transform(X)
    pca        = PCA(n_components=2, random_state=42)
    X_pca      = pca.fit_transform(X_scaled)
    sample_idx = np.random.choice(len(X), min(800, len(X)), replace=False)

    STATE.update({
        'X': X, 'y': y,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test.tolist(),
        'results':   results,
        'best_name': best_name,
        'X_pca':     X_pca[sample_idx],
        'y_pca':     y[sample_idx].tolist(),
        'pca_var':   pca.explained_variance_ratio_.tolist(),
        'scaler_pca': scaler_pca,
    })
    print(f"✅  Done — best: {best_name}  acc={results[best_name]['accuracy']}")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── Overview ─────────────────────────────────
@app.route('/api/overview')
def api_overview():
    results   = STATE['results']
    best_name = STATE['best_name']
    y         = STATE['y']

    table = []
    for name, r in results.items():
        table.append({
            'name':    name,
            'accuracy': r['accuracy'],
            'f1':       r['f1'],
            'cv_mean':  r['cv_mean'] or '—',
            'cv_std':   r['cv_std'] or '—',
            'is_best':  name == best_name,
        })

    dist = {EMOTION_NAMES[i]: int(np.sum(y == i)) for i in range(N_CLASSES)}

    return jsonify({
        'n_samples':   int(len(y)),
        'n_features':  int(STATE['X'].shape[1]),
        'n_classes':   N_CLASSES,
        'emotions':    EMOTION_NAMES,
        'emo_colors':  EMO_COLORS,
        'distribution': dist,
        'best_name':   best_name,
        'best_acc':    results[best_name]['accuracy'],
        'best_f1':     results[best_name]['f1'],
        'table':       table,
    })

# ── Performance ───────────────────────────────
@app.route('/api/performance/<model_name>')
def api_performance(model_name):
    results = STATE['results']
    y_test  = STATE['y_test']

    if model_name not in results:
        return jsonify({'error': 'Model not found'}), 404

    res    = results[model_name]
    y_pred = res['y_pred']

    # Compute CV lazily and cache
    if res['cv_mean'] is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            res['pipeline'], STATE['X_train'], STATE['y_train'],
            cv=cv, scoring='accuracy', n_jobs=1
        )
        res['cv_mean'] = round(float(cv_scores.mean()), 4)
        res['cv_std']  = round(float(cv_scores.std()), 4)
        res['cv_all']  = cv_scores.round(4).tolist()
        # fill other models with same placeholder array for chart
    cv_boxplot = {}
    for name, r in results.items():
        cv_boxplot[name] = r['cv_all'] if r['cv_all'] else [r['accuracy']] * 3

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Recall per class
    cm_np  = np.array(cm)
    recall = (cm_np.diagonal() / cm_np.sum(axis=1)).round(4).tolist()

    # F1 per class
    f1_per = []
    for i in range(N_CLASSES):
        yt = (np.array(y_test) == i).astype(int)
        yp = (np.array(y_pred) == i).astype(int)
        f1_per.append(round(float(f1_score(yt, yp, zero_division=0)), 4))

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=EMOTION_NAMES, output_dict=True
    )

    # CV boxplot data
    cv_data = {}
    for name, r in results.items():
        cv_data[name] = r['cv_all']

    return jsonify({
        'model_name':      model_name,
        'accuracy':        res['accuracy'],
        'f1':              res['f1'],
        'cv_mean':         res['cv_mean'] or res['accuracy'],
        'cv_std':          res['cv_std'] or 0.0,
        'confusion_matrix': cm,
        'recall_per_class': recall,
        'f1_per_class':    f1_per,
        'report':          report,
        'cv_boxplot':      cv_boxplot,
        'emotions':        EMOTION_NAMES,
        'emo_colors':      EMO_COLORS,
    })

# ── PCA ───────────────────────────────────────
@app.route('/api/pca')
def api_pca():
    X_pca = STATE['X_pca']
    y_pca = STATE['y_pca']
    var   = STATE['pca_var']

    points = []
    for i in range(N_CLASSES):
        mask = np.array(y_pca) == i
        xs   = X_pca[mask, 0].round(4).tolist()
        ys   = X_pca[mask, 1].round(4).tolist()
        points.append({'emotion': EMOTION_NAMES[i], 'color': EMO_COLORS[i], 'x': xs, 'y': ys})

    return jsonify({
        'points':     points,
        'var_pc1':    round(var[0], 4),
        'var_pc2':    round(var[1], 4),
        'total_var':  round(sum(var), 4),
    })

# ── MFCC Profiles ─────────────────────────────
@app.route('/api/mfcc_profiles')
def api_mfcc_profiles():
    X_test = STATE['X_test']
    y_test = STATE['y_test']

    profiles = []
    for i in range(N_CLASSES):
        idx = [j for j, v in enumerate(y_test) if v == i]
        if not idx:
            profiles.append({'emotion': EMOTION_NAMES[i], 'values': []})
            continue
        feat = X_test[idx[0], :N_MFCC].round(4).tolist()
        profiles.append({
            'emotion': EMOTION_NAMES[i],
            'color':   EMO_COLORS[i],
            'values':  feat,
        })
    return jsonify({'profiles': profiles, 'n_mfcc': N_MFCC})

# ── Predict (manual input) ────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data       = request.get_json()
    model_name = data.get('model', STATE['best_name'])
    emotion_id = int(data.get('emotion_id', 0))
    noise      = float(data.get('noise', 0.12))

    if model_name not in STATE['results']:
        return jsonify({'error': 'Model not found'}), 404

    feat     = simulate_features(emotion_id, noise=noise).reshape(1, -1)
    pipeline = STATE['results'][model_name]['pipeline']
    pred     = int(pipeline.predict(feat)[0])
    proba    = pipeline.predict_proba(feat)[0].round(4).tolist()

    return jsonify({
        'true_emotion':      EMOTION_NAMES[emotion_id],
        'predicted_emotion': EMOTION_NAMES[pred],
        'correct':           pred == emotion_id,
        'probabilities':     proba,
        'emotions':          EMOTION_NAMES,
        'emo_colors':        EMO_COLORS,
    })

# ── Models list ───────────────────────────────
@app.route('/api/models')
def api_models():
    return jsonify({
        'models':    list(STATE['results'].keys()),
        'best_name': STATE['best_name'],
    })

# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀  Training models …")
    train()
    print("🌐  http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)