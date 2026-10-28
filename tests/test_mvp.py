"""Verifica separación de datos, inferencia persistida, abstención y minimización."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset import ROOT, digest, read_corpus
from src.model import append_audit, classify, load_model


class MVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.metadata = load_model()
        cls.rows = read_corpus()

    def test_scenario_and_text_disjointness(self):
        for a, b in [('train', 'test'), ('train', 'validation'), ('test', 'validation')]:
            for key in ('group', 'text'):
                self.assertFalse({r[key] for r in self.rows if r['split'] == a} &
                                 {r[key] for r in self.rows if r['split'] == b})

    def test_training_vocabulary_does_not_include_test_only_word(self):
        tokenizer = self.pipeline.named_steps['tfidf'].build_tokenizer()
        train = set(t for r in self.rows if r['split'] == 'train' for t in tokenizer(r['text'].lower()))
        test = set(t for r in self.rows if r['split'] == 'test' for t in tokenizer(r['text'].lower()))
        unseen = test - train
        self.assertTrue(unseen)
        self.assertFalse(unseen & set(self.pipeline.named_steps['tfidf'].vocabulary_))

    def test_round_trip_prediction(self):
        other, metadata = load_model()
        text = 'El empleador no paga el salario del contrato de trabajo.'
        self.assertEqual(classify(other, text, metadata), classify(self.pipeline, text, self.metadata))

    def test_examples_represent_four_categories(self):
        examples = {'laboral': 'El empleador no paga el salario ni las cesantías.',
                    'familia': 'El padre solicita la custodia de su hijo y una cuota de alimentos.',
                    'civil': 'El propietario reclama los cánones del contrato de arrendamiento.',
                    'penal': 'La Fiscalía investiga una estafa con suplantación de identidad.'}
        for label, text in examples.items():
            with self.subTest(label=label):
                self.assertEqual(classify(self.pipeline, text, self.metadata)['suggested_class'], label)

    def test_invalid_inputs(self):
        for text in ('', '  ', None, 'a' * 5001):
            with self.subTest(text=str(text)[:10]), self.assertRaises(ValueError):
                classify(self.pipeline, text, self.metadata)

    def test_unseen_and_short_queries_abstain(self):
        for text in ('zxqv lmnpr qrstuv', 'Ayuda', 'Dame una receta de pastel de chocolate.'):
            self.assertEqual(classify(self.pipeline, text, self.metadata)['decision'], 'revision_humana')

    def test_sensitive_query_is_escalated(self):
        result = classify(self.pipeline, 'Mi pareja me amenaza y controla los alimentos de mi hijo.', self.metadata)
        self.assertEqual(result['decision'], 'revision_humana')
        self.assertIn('senal_sensible_revision_prioritaria', result['reasons'])

    def test_explanation_reconstructs_logit(self):
        text = 'Mi empleador no paga el salario del contrato de trabajo.'
        result = classify(self.pipeline, text, self.metadata)
        x = self.pipeline.named_steps['tfidf'].transform([text])
        clf = self.pipeline.named_steps['classifier']
        index = list(clf.classes_).index(result['suggested_class'])
        logit = float((x @ clf.coef_[index])[0] + clf.intercept_[index])
        self.assertAlmostEqual(logit, result['full_logit'], places=5)
        self.assertTrue(result['requires_human_review'])
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1, places=5)

    def test_metadata_hash_matches_current_artifact(self):
        self.assertEqual(digest(ROOT / 'models/classifier.joblib'), self.metadata['model_sha256'])
        self.assertEqual(digest(ROOT / 'data/corpus.csv'), self.metadata['corpus_sha256'])

    def test_audit_excludes_query_and_features(self):
        result = classify(self.pipeline, 'El empleador no paga el salario.', self.metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.jsonl'
            append_audit(result, path)
            record = json.loads(path.read_text(encoding='utf-8'))
        self.assertNotIn('text', record)
        self.assertNotIn('contributions', record)
        self.assertNotIn('empleador', json.dumps(record))
        self.assertEqual(record['version'], self.metadata['version'])


if __name__ == '__main__':
    unittest.main()
