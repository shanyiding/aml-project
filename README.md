\# AML Explainable Risk Scoring Project



End-to-end AML risk scoring pipeline combining:

\- Rule-based signals

\- Supervised ML

\- Unsupervised anomaly detection

\- Explainability (SHAP + reason codes)



\## Structure

\- `data/` – raw \& processed data (not committed)

\- `knowledge\_base/` – AML rules \& mappings

\- `src/` – pipelines, features, models

\- `notebooks/` – exploration only

\- `docs/` – methodology \& documentation



\## Reproducibility

```bash

python src/pipeline/run\_all.py



