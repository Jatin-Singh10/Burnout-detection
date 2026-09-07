# AI-Based Early Burnout Detection — Deployment Package

This folder contains everything needed to deploy your trained model as a live,
interactive web app.

## What's inside
- `app.py` — Streamlit dashboard (student intake form → risk score, risk tier,
  recommendations, model transparency panel).
- `burnout_model.joblib` — the trained, calibrated Random Forest pipeline
  (preprocessing + model bundled together) exported from your notebook.
- `metadata.json` — feature list, categorical option lists, numeric ranges,
  decision threshold, and held-out test metrics, used by the app.
- `requirements.txt` — exact pinned dependency versions matching the
  environment the model was trained in (important: mismatched scikit-learn/
  pandas/numpy versions can break `joblib.load`).

## Test it locally first
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the printed local URL (usually http://localhost:8501).

---

## Option 1 — Streamlit Community Cloud (free, easiest, recommended)
1. Create a new GitHub repository and push this folder's contents to it
   (`app.py`, `burnout_model.joblib`, `metadata.json`, `requirements.txt`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the main file to `app.py`.
4. Click **Deploy**. You'll get a public URL like
   `https://your-app-name.streamlit.app` within a couple of minutes.
5. Any future `git push` to the repo automatically redeploys the app.

No server management, free tier is sufficient for a college project/demo.

## Option 2 — Hugging Face Spaces (free, good if you want more control)
1. Create a free account at https://huggingface.co.
2. Create a new **Space** → SDK: **Streamlit**.
3. Upload the same four files (or push via `git` — Spaces are git repos).
4. The Space builds automatically and gives you a public URL
   (`https://huggingface.co/spaces/<username>/<space-name>`).

## Option 3 — Render.com (free/low-cost, if you outgrow Streamlit-only hosting)
1. Push this folder to a GitHub repo.
2. On https://render.com, create a **New Web Service** from that repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy — Render gives you a public HTTPS URL.

## Option 4 — Docker (for AWS / GCP / Azure / any VPS)
A `Dockerfile` is included. Build and run:
```bash
docker build -t burnout-app .
docker run -p 8501:8501 burnout-app
```
Push the image to any container registry (ECR, GCR, Docker Hub) and deploy it
on AWS App Runner / ECS, GCP Cloud Run, or Azure Container Apps for a
production-grade, scalable deployment.

---

## Notes on the model itself
- Algorithm: **Logistic Regression** (selected over Random Forest via
  validation-set F2/PR-AUC comparison — near-identical accuracy at a
  fraction of the file size), Platt-calibrated.
- Held-out test performance: see `metadata.json → final_test_metrics`
  (Recall ≈ 0.96, ROC-AUC ≈ 0.92, PR-AUC ≈ 0.94 on this run).
- Decision threshold is tuned for **recall** (catching at-risk students),
  which trades off some precision — appropriate for an early-warning tool,
  but worth stating explicitly in any report or demo.
- This is a research/decision-support prototype trained on a public survey
  dataset — it is **not** a clinical or diagnostic tool. The app footer
  includes this disclaimer; keep it in any deployment.

## Regenerating the model
If you want to retrain (e.g., with the full notebook's hyperparameter search,
more models, or SHAP/LIME explanations wired into the app), rerun your
notebook end-to-end and re-export with:
```python
import joblib
joblib.dump(calibrated_model, "burnout_model.joblib")
```
then update `metadata.json` accordingly and redeploy.
