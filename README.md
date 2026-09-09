# Insurance & App Fraud Detection

A Flask web application that detects fraud in two independent domains:

1. **Vehicle insurance claims** — a trained Random Forest classifies a claim as fraudulent or genuine from 14 claim attributes.
2. **Google Play Store apps** — reviews for any app are scraped live and run through NLTK VADER sentiment analysis to flag suspicious apps.

Both modules ship with extensive exploratory data visualisation, rendered server-side with matplotlib/seaborn and streamed to the browser as inline base64 images.

Built by SY CS (AIML) students at Vishwakarma Institute of Technology, Pune.

---

## Features

### Vehicle insurance fraud
- Form-driven prediction over 14 claim features
- Pre-trained Random Forest (`RFModel.pkl`) with one-hot encoding aligned to the training schema
- 12-chart exploratory analysis dashboard (distributions, correlation heatmap, NetworkX brand/fraud graph, hexbin, violin plots)
- Browsable dataset view with 10 summary charts

### Play Store app fraud
- Scrapes *all* reviews for a given app ID via `google-play-scraper`
- Text pipeline: lowercase → punctuation strip → tokenise → stopword removal → WordNet lemmatisation
- VADER sentiment scoring, cross-checked against the star rating
- Verdict plus 6 charts: word cloud, top proper nouns, sentiment/length heatmap, rating distribution

---

## Tech stack

| Layer | Tools |
|---|---|
| Web | Flask, Jinja2, vanilla CSS/JS |
| ML | scikit-learn (Random Forest), joblib |
| NLP | NLTK (VADER, WordNet, punkt, stopwords) |
| Data | pandas |
| Charts | matplotlib, seaborn, wordcloud, networkx |
| Scraping | google-play-scraper |
| Deploy | Docker |

---

## Project structure

```
.
├── app.py                      # Entire backend: routes, inference, all chart generation
├── RFModel.pkl                 # Pre-trained Random Forest (41 MB, joblib)
├── DVCarFraudDetection.csv     # Claims dataset — 16,184 rows × 15 columns
├── X_train.csv                 # Training matrix; used only for its 24 column names
├── requirements.txt
├── Dockerfile
├── templates/                  # 9 Jinja templates
│   ├── index.html              # Landing page
│   ├── vehicle.html            # Insurance claim form
│   ├── prediction_result.html  # Insurance verdict + 8 charts
│   ├── fraudapp.html           # App ID form
│   ├── app_result.html         # App verdict + 6 charts
│   ├── dataset.html            # Dataset browser + 10 charts
│   ├── insurance_analysis.html # 12-chart EDA dashboard
│   └── app_analysis*.html      # App review analysis
└── static/                     # Per-page stylesheets and images
```

---

## Getting started

### Prerequisites

**Python 3.9 – 3.11.** `requirements.txt` pins `numpy==1.25.2` and `scikit-learn==1.2.2`, neither of which builds on 3.12+. The scikit-learn pin is deliberate: `RFModel.pkl` is a pickle and will warn or fail to load on a different version.

### Install

```bash
git clone https://github.com/KanadK/vehicle.git
cd vehicle
python -m venv env
source env/bin/activate      # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Then open <http://127.0.0.1:5000>.

On first start the app downloads the four NLTK corpora it needs (`punkt_tab`, `stopwords`, `wordnet`, `vader_lexicon`, roughly 15 MB) into your local `nltk_data`. This happens once — subsequent starts make no network calls.

### Docker

```bash
docker build -t fraud-detection .
docker run -p 5000:5000 fraud-detection
```

The container serves through gunicorn (2 sync workers, 120s timeout) rather than the Flask development server. `python app.py` remains the local development entry point.

---

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Landing page |
| `/vehicle_insurance` | GET | Insurance claim form |
| `/predict/insurance` | GET | Insurance claim form |
| `/predict/insurance` | POST | Run the model, return verdict + 8 charts |
| `/mobile_app` | GET | App ID form |
| `/predict/app` | GET | App ID form |
| `/predict/app` | POST | Scrape reviews, return verdict + 6 charts |
| `/analysis/insurance` | GET | 12-chart EDA dashboard |
| `/analysis/app` | GET / POST | App review analysis form and results |
| `/dataset` | GET | Dataset browser + 10 charts |

---

## How it works

### Insurance prediction

The model was trained on a one-hot encoded 24-column matrix. A single submitted form row cannot reproduce those columns on its own, so the handler realigns it:

```python
processed_user_input = pd.get_dummies(user_df)
processed_user_input = processed_user_input.reindex(
    columns=X_train.columns, fill_value=0
)
```

`get_dummies` on one row emits only the categories that row happens to contain (e.g. `CarCategory_Sedan`). The `reindex` restores all 24 features in exact training order and zero-fills the rest. Without it the model would receive mismatched features and predict nonsense.

### App sentiment

A review counts as `positive` only when VADER agrees with the star rating:

```python
if sentiment_score >= 0.05 and score >= 3:   return 'positive'
elif sentiment_score <= -0.05 and score < 3: return 'negative'
else:                                        return 'neutral'
```

Requiring both signals to agree discards reviews whose text contradicts their rating. The final verdict is a plain majority: more positives than negatives means the app is not flagged.

### Charts

No image files are ever written. Every figure is saved to an in-memory buffer, base64-encoded, and embedded directly in the page:

```python
buffer = io.BytesIO()
fig.savefig(buffer, format='png')
base64.b64encode(buffer.getvalue()).decode()
```

---

## Dataset

`DVCarFraudDetection.csv` — 16,184 claims, 15 columns.

| Feature | Type | Values |
|---|---|---|
| `CarCompany` | categorical | 18 manufacturers |
| `CarCategory` | categorical | Sedan, Sport, Utility |
| `BasePolicy` | categorical | All Perils, Collision, Liability |
| `AccidentArea` | categorical | Urban, Rural |
| `Fault` | categorical | Policy Holder, Third Party |
| `AgentType` | categorical | External, Internal |
| `PoliceReportFiled`, `WitnessPresent`, `IsAddressChanged`, `OwnerGender` | binary | — |
| `OwnerAge` | numeric | 21 – 73 |
| `CarPrice` | numeric | 10,008 – 56,761 |
| `NumberOfSuppliments` | numeric | 0 – 7 |
| `PastNumberOfClaims` | numeric | 0 – 6 |
| `IsFraud` | target | 0 / 1 |

The classes are close to balanced — 7,571 fraudulent against 8,613 genuine, a 46.8% fraud rate. Real insurance fraud runs in the low single digits, so this dataset has been resampled or synthesised. Accuracy figures from it will not transfer to production data.

---

## Known limitations

- **Training code is not in this repository** — only the serialised `RFModel.pkl`. No accuracy metrics are published here.
- **`RFModel.pkl` (41 MB) is committed to git**, which makes clones slow, and it unpickles only under scikit-learn 1.2.2.
- **Prediction quality has not been re-measured** since the `CarPrice` scaling fix. The change is correct with respect to the training data, but no accuracy figure has been recomputed against the model.
- **`requirements.txt` lists `gunicorn` twice.**

---

## Credits

SY CS (AIML), Vishwakarma Institute of Technology, Pune.
