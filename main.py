"""Interfaz de consola: entrenamiento, evaluación, demostración y consultas nuevas."""

import argparse
import json
import sys
from datetime import datetime, timezone

from src.dataset import ROOT, generate, read_corpus
from src.model import append_audit, classify, load_model, train


def main():
    parser = argparse.ArgumentParser(description='JustIA: clasificación jurídica temática con revisión humana.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('train', help='Genera corpus, entrena y guarda modelo con metadatos.')
    sub.add_parser('evaluate', help='Evalúa el modelo guardado, sin reentrenar.')
    sub.add_parser('demo', help='Ejecuta el ciclo completo y guarda salidas de ejemplo.')
    predict = sub.add_parser('predict', help='Clasifica texto nuevo usando el modelo persistido.')
    predict.add_argument('--text', required=True)
    predict.add_argument('--audit', action='store_true', help='Registra solo versión y decisión, sin texto personal.')
    args = parser.parse_args()
    try:
        if args.command in ('train', 'demo'):
            rows = generate()
            pipeline, metadata = train(rows)
        else:
            pipeline, metadata = load_model()
        if args.command == 'train':
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        elif args.command == 'predict':
            result = classify(pipeline, args.text, metadata)
            if args.audit:
                append_audit(result, ROOT / 'runtime/audit.jsonl')
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            from src.evaluate import evaluate
            metrics = evaluate(pipeline, metadata, read_corpus())
            if args.command == 'demo':
                cases = ['Mi empleador no pagó el salario ni las cesantías de mi contrato de trabajo.',
                         'Quiero acordar la custodia y los alimentos de mi hijo menor.',
                         'Dame una receta de pastel de chocolate.',
                         'Mi pareja me amenaza y controla el dinero para los alimentos de mi hijo.']
                examples = [dict(text=text, **classify(pipeline, text, metadata)) for text in cases]
                (ROOT / 'resultados/examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
                lines = ['JUSTIA | EJECUCION REAL DEL MVP', 'UTC: ' + datetime.now(timezone.utc).isoformat(),
                         'Modelo: ' + metadata['version'], 'Corpus: 192 filas; train=120; validacion=36; prueba=36',
                         f"Prueba: accuracy={metrics['test']['accuracy']:.4f}; F1 macro={metrics['test']['f1_macro']:.4f}",
                         f"Cobertura con abstencion: {metrics['test']['accepted']}/36", '']
                for item in examples:
                    lines.extend([item['text'], f"Salida: {item['decision']} | puntaje={item['score']:.3f}",
                                  'Terminos: ' + ', '.join(x['term'] for x in item['contributions']), ''])
                transcript = '\n'.join(lines)
                (ROOT / 'evidencias/demo_console.txt').write_text(transcript, encoding='utf-8')
                print(transcript)
            else:
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
