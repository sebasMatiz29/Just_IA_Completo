# JustIA | Proyecto final

MVP de clasificación temática jurídica: civil, familia, laboral y penal. Usa TF-IDF y regresión logística de scikit-learn. Entrena, guarda el modelo, evalúa y clasifica nuevas consultas mediante consola. No necesita nube, API key ni GPU.

## Ejecución

Probado con Python 3.12. Instalar en un entorno virtual desde la raíz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py demo
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe main.py predict --text "Mi empleador no pagó el salario ni las cesantías."
```

Comandos separados: `python main.py train`, `python main.py evaluate` y `python main.py predict --text "consulta"`. `demo` regenera los datos y reemplaza el artefacto y resultados académicos con una nueva ejecución. `evaluate` no reentrena. Entrenar después de modificar el código o los escenarios exige regenerar las evidencias y revisar el informe.

## Archivos entregados

- `data/scenarios.json`: 64 escenarios sintéticos originales, 16 por tema.
- `data/corpus.csv`: 192 filas; 120 entrenamiento, 36 validación, 36 prueba. Variantes de un escenario siempre permanecen en la misma partición.
- `data/challenge.json`: 12 consultas adicionales y cuatro entradas fuera de dominio o insuficientes; evaluación exploratoria.
- `src/`: generación, entrenamiento, inferencia, explicación y evaluación.
- `models/`: modelo efectivamente entrenado y metadatos con SHA-256 del corpus y artefacto.
- `resultados/`: métricas, predicciones individuales, ejemplos y comprobaciones.
- `evidencias/`: matriz de confusión, salida de consola y pruebas.
- `tests/`: pruebas de particiones, vocabulario, persistencia, explicación, entradas y auditoría.
- `Docs/`: informe institucional, reflexión, guía y comprimido para entregar. Se excluye de Git, no de la entrega académica.

## Interpretación y límites

Consultar `resultados/metrics.json` para resultados exactos. El puntaje de `predict_proba` no está calibrado y no es certeza jurídica. La política usa umbrales heurísticos previos (0,45 de puntaje, margen 0,12, cobertura léxica 0,35 y cuatro tokens); no se optimizaron contra la prueba. La señal de amenazas/violencia deriva prioritariamente y no es un detector exhaustivo. Incluso las categorías aceptadas requieren revisión humana. Una consulta puede pertenecer a varias áreas; este MVP es monoclase.

Los ejemplos son sintéticos y no están validados por un experto jurídico; no contienen expedientes de personas. Las tres variantes por escenario están correlacionadas. Los resultados no prueban generalización, imparcialidad ni aptitud productiva. El aprendizaje supervisado permite imprecisiones y se conservan los errores en los CSV de salida.

## Trazabilidad y privacidad

`predict --audit` añade un registro mínimo en `runtime/audit.jsonl`: versión, huellas del modelo/corpus, decisión, puntaje y motivos. No guarda la pregunta ni sus términos. La consola muestra contribuciones léxicas, que podrían contener datos personales si se ingresaran consultas reales; esta demostración debe utilizar solo datos sintéticos. `runtime/` se excluye de Git y del paquete.

El archivo joblib solo debe cargarse si fue generado localmente o se confía en su procedencia. La comprobación de hash detecta alteraciones respecto a los metadatos, pero no autentica un proveedor. La persistencia requiere la misma versión de scikit-learn.

## Diferencia respecto a los casos anteriores

Se conserva el contexto JustIA y sus cuatro categorías. El proyecto final elige UN reto permitido y entrega un ciclo completo con modelo persistido, clasificación nueva, separación por escenarios, explicación lineal, abstención, auditoría minimizada y evaluación reproducible. No presenta los resultados anteriores de Naive Bayes como si fueran del modelo actual ni usa controles simulados como evidencia de entrenamiento.

No se ha publicado este proyecto en GitHub. Puede entregarse el ZIP en Docs o publicarse después en un repositorio propio.
