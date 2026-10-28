"""Ejecuta pruebas y conserva una transcripción reproducible sin editar resultados."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
(ROOT / 'evidencias').mkdir(exist_ok=True)
(ROOT / 'evidencias/tests_console.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
(ROOT / 'resultados/verification.json').write_text(json.dumps(dict(
    timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
    platform=platform.platform(), command='python -m unittest discover -s tests -v',
    returncode=result.returncode), indent=2), encoding='utf-8')
print(result.stdout + result.stderr)
sys.exit(result.returncode)
