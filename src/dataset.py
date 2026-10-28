"""Construye particiones reproducibles sin compartir escenarios entre conjuntos."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """Identifica exactamente los bytes del corpus o del modelo utilizado."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(root=ROOT):
    scenarios = json.loads((root / 'data/scenarios.json').read_text(encoding='utf-8'))
    rows = []
    for label, texts in scenarios.items():
        for index, text in enumerate(texts):
            # Las tres variantes permanecen juntas; no hay reparto aleatorio por fila.
            split = 'train' if index < 10 else 'validation' if index < 13 else 'test'
            group = f'{label}-{index + 1:02}'
            for variant, prefix in enumerate(('', 'Solicito orientación: ', 'Consulta recibida: ')):
                rows.append(dict(id=f'{group}-{variant}', group=group, split=split,
                                 text=prefix + text, label=label, source='sintetico_autoria_academica'))
    path = root / 'data/corpus.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_corpus(root=ROOT):
    with (root / 'data/corpus.csv').open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
