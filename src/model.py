"""Entrena el pipeline, conserva su procedencia y explica predicciones lineales."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .dataset import ROOT, digest

VERSION = 'justia-cls-1.0.0'
POLICY = {'min_tokens': 4, 'min_coverage': 0.35, 'min_score': 0.45, 'min_margin': 0.12}
WARNING = 'Clasificación temática experimental. No es asesoría ni decisión jurídica; requiere revisión humana.'


def make_pipeline():
    # El vocabulario se aprende exclusivamente con entrenamiento mediante Pipeline.fit.
    return Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
        ('classifier', LogisticRegression(C=4.0, max_iter=1000, random_state=42)),
    ])


def train(rows, root=ROOT):
    training = [r for r in rows if r['split'] == 'train']
    pipeline = make_pipeline()
    pipeline.fit([r['text'] for r in training], [r['label'] for r in training])
    directory = root / 'models'
    directory.mkdir(exist_ok=True)
    artifact = directory / 'classifier.joblib'
    joblib.dump(pipeline, artifact)
    metadata = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
                    sklearn_version=sklearn.__version__, corpus_sha256=digest(root / 'data/corpus.csv'),
                    scenarios_sha256=digest(root / 'data/scenarios.json'), model_sha256=digest(artifact),
                    labels=pipeline.classes_.tolist(), train_rows=len(training),
                    policy=POLICY, policy_status='umbrales heurísticos fijados antes de evaluar; no calibrados',
                    purpose='clasificacion_tematica', deployment='solo_demostracion_academica', seed=42)
    (directory / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return pipeline, metadata


def load_model(root=ROOT):
    """Carga solo el artefacto local confiable; joblib no debe abrir modelos desconocidos."""
    metadata = json.loads((root / 'models/metadata.json').read_text(encoding='utf-8'))
    path = root / 'models/classifier.joblib'
    if digest(path) != metadata['model_sha256']:
        raise ValueError('La huella del modelo no coincide con metadata.json.')
    if sklearn.__version__ != metadata['sklearn_version']:
        raise ValueError('Use la versión de scikit-learn registrada en requirements.txt.')
    return joblib.load(path), metadata


def classify(pipeline, text, metadata):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('La consulta debe contener texto.')
    if len(text) > 5000:
        raise ValueError('La consulta supera el límite académico de 5000 caracteres.')
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    vector = vectorizer.transform([text])
    scores = classifier.predict_proba(vector)[0]
    order = np.argsort(scores)[::-1]
    winner = int(order[0])
    tokens = vectorizer.build_tokenizer()(text.lower())
    coverage = sum(t in vectorizer.vocabulary_ for t in tokens) / max(len(tokens), 1)
    margin = float(scores[order[0]] - scores[order[1]])
    policy = metadata['policy']
    reasons = []
    # Señal preventiva acotada: no diagnostica violencia ni sustituye atención profesional.
    if re.search(r'\b(amenaz\w*|violen\w*|agresi\w*|suicid\w*|matar\w*|arma\w*)\b', text.lower()):
        reasons.append('senal_sensible_revision_prioritaria')
    if len(tokens) < policy['min_tokens']:
        reasons.append('texto_insuficiente')
    if coverage < policy['min_coverage']:
        reasons.append('vocabulario_poco_cubierto')
    if float(scores[winner]) < policy['min_score']:
        reasons.append('puntaje_bajo')
    if margin < policy['min_margin']:
        reasons.append('categorias_cercanas')
    # x_j * w_kj explica el logit lineal, no causalidad ni razonamiento jurídico.
    contributions = vector.toarray()[0] * classifier.coef_[winner]
    names = vectorizer.get_feature_names_out()
    positive = [i for i in np.argsort(contributions)[::-1] if contributions[i] > 0][:5]
    return dict(version=metadata['version'], model_sha256=metadata['model_sha256'],
                corpus_sha256=metadata['corpus_sha256'],
                suggested_class=str(classifier.classes_[winner]),
                decision='revision_humana' if reasons else str(classifier.classes_[winner]),
                score=round(float(scores[winner]), 6), margin=round(margin, 6),
                lexical_coverage=round(coverage, 6), reasons=reasons,
                probabilities={str(label): round(float(score), 6) for label, score in zip(classifier.classes_, scores)},
                contributions=[dict(term=str(names[i]), weight=round(float(contributions[i]), 6)) for i in positive],
                intercept=round(float(classifier.intercept_[winner]), 6),
                full_logit=round(float(classifier.decision_function(vector)[0, winner]), 6),
                explanation='Contribuciones positivas al logit de la clase sugerida; puntaje no calibrado.',
                requires_human_review=True, warning=WARNING)


def append_audit(result, path):
    """Guarda versión y decisión sin texto, tokens, identidad ni hashes de la consulta."""
    record = {key: result[key] for key in ('version', 'model_sha256', 'corpus_sha256',
                                          'decision', 'score', 'reasons', 'requires_human_review')}
    record['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
