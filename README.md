@"
# Loan Default Predictions — How to View the Dashboard

You can view the dashboard in two ways.

## Option A: Open the hosted link (easiest)
Open this link in your browser:
https://predictingloandefault.streamlit.app/

---

## Option B: Run locally (Windows / macOS / Linux)

### What you need
- Python 3.9 or newer
- Internet connection (first run only, to install packages)

### Steps
1) Download the project (ZIP) or clone the repo, then open a terminal in the project folder.

2) Create a virtual environment:

**Windows (PowerShell):**
python -m venv .venv
.venv\Scripts\Activate.ps1

**macOS / Linux:**
python3 -m venv .venv
source .venv/bin/activate

3) Install requirements:
pip install -r requirements.txt

4) Add a dataset (CSV):
Place your CSV in one of these paths so the app can find it:
data/clean_credit.csv
data/credit_clean.csv
data/credit.csv
cs-training.csv
train.csv

5) Run the app:
streamlit run dashboard.py

If the app doesn’t update, use the Streamlit menu: ☰ → Clear cache → Rerun.
"@ | Set-Content -Encoding UTF8 README.md
