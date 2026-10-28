"""Mide por separado clasificación forzada y cobertura con abstención."""

import csv
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

from .dataset import ROOT
from .model import classify


def evaluate(pipeline, metadata, rows, root=ROOT):
    output = root / 'resultados'
    figures = root / 'evidencias'
    output.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    all_results = {}
    for split in ('validation', 'test'):
        subset = [r for r in rows if r['split'] == split]
        predictions = [classify(pipeline, r['text'], metadata) for r in subset]
        truth = [r['label'] for r in subset]
        forced = [p['suggested_class'] for p in predictions]
        accepted = [i for i, p in enumerate(predictions) if p['decision'] != 'revision_humana']
        result = dict(rows=len(subset), groups=len({r['group'] for r in subset}),
                      accuracy=float(accuracy_score(truth, forced)),
                      f1_macro=float(f1_score(truth, forced, average='macro')),
                      per_class=classification_report(truth, forced, output_dict=True, zero_division=0),
                      coverage=len(accepted) / len(subset), accepted=len(accepted),
                      accepted_accuracy=float(accuracy_score([truth[i] for i in accepted], [forced[i] for i in accepted])) if accepted else None,
                      confusion_matrix=confusion_matrix(truth, forced, labels=metadata['labels']).tolist())
        all_results[split] = result
        with (output / f'predictions_{split}.csv').open('w', encoding='utf-8', newline='') as handle:
            fields = ['id', 'group', 'text', 'expected', 'predicted', 'decision', 'score', 'correct']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row, prediction in zip(subset, predictions):
                writer.writerow(dict(id=row['id'], group=row['group'], text=row['text'], expected=row['label'],
                                     predicted=prediction['suggested_class'], decision=prediction['decision'],
                                     score=prediction['score'], correct=row['label'] == prediction['suggested_class']))
    train_y = [r['label'] for r in rows if r['split'] == 'train']
    test_y = [r['label'] for r in rows if r['split'] == 'test']
    dummy = DummyClassifier(strategy='most_frequent').fit([[0]] * len(train_y), train_y)
    dummy_predictions = dummy.predict([[0]] * len(test_y))
    all_results['baseline'] = dict(accuracy=float(accuracy_score(test_y, dummy_predictions)),
                                   f1_macro=float(f1_score(test_y, dummy_predictions, average='macro')))
    challenges = json.loads((root / 'data/challenge.json').read_text(encoding='utf-8'))
    challenge_outputs = [dict(**case, result=classify(pipeline, case['text'], metadata)) for case in challenges]
    (output / 'challenge_predictions.json').write_text(json.dumps(challenge_outputs, ensure_ascii=False, indent=2), encoding='utf-8')
    known = [r for r in challenge_outputs if r['expected'] != 'revision_humana']
    unknown = [r for r in challenge_outputs if r['expected'] == 'revision_humana']
    all_results['challenge'] = dict(known_count=len(known),
        forced_accuracy=sum(r['expected'] == r['result']['suggested_class'] for r in known) / len(known),
        covered_count=sum(r['result']['decision'] != 'revision_humana' for r in known),
        ood_count=len(unknown), ood_abstained=sum(r['result']['decision'] == 'revision_humana' for r in unknown))
    all_results['scope'] = 'Corpus sintético, escenarios separados; no es validación externa ni garantía jurídica.'
    all_results['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    all_results['model_sha256'] = metadata['model_sha256']
    (output / 'metrics.json').write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7, 4.4), layout='constrained')
    ConfusionMatrixDisplay(np.array(all_results['test']['confusion_matrix']), display_labels=metadata['labels']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set(title='Prueba reservada por escenarios', xlabel='Clase predicha', ylabel='Clase de referencia')
    fig.savefig(figures / 'figura_1_matriz.png', dpi=180)
    plt.close(fig)
    return all_results
