# E-commerce delivery delay analysis (Olist)

SVU – Programming for Data Science (DPD601), F25 course project.

We clean and join the seven tables of the Brazilian Olist e-commerce dataset, study which order
characteristics go together with late delivery, and train models that estimate, **at the moment an
order is approved**, whether it will reach the customer after the estimated delivery day.

## Data source and licence

Olist, & Sionek, A. (2018). *Brazilian E-Commerce Public Dataset by Olist*. Kaggle.
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce — licence **CC BY-NC-SA 4.0**
(attribution, non-commercial use, share alike). The data are historical (2016–2018, Brazil); the
results should not be generalised directly to other markets.

Only the 7 tables of the course pack are used (geolocation and reviews are not used):
`olist_orders`, `olist_customers`, `olist_order_items`, `olist_order_payments`, `olist_products`,
`olist_sellers`, `product_category_name_translation`. The raw files are never modified.

## Tools and versions

Python 3.11, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.5.2, SciPy 1.14.1, Matplotlib 3.9.2,
seaborn 0.13.2, pytest 8.3.3 (pinned in `requirements.txt`). Graphviz is optional (only to
re-draw the ER diagram). Git for version control.

## Folder structure

```
ecommerce_delivery/
├── README.md, requirements.txt, run_all.sh
├── data/
│   ├── raw/                 # the 7 original CSV files (not in git, see "Setup")
│   └── processed/           # orders_full.csv (all orders), orders_model.csv (modelling table)
├── src/
│   ├── config.py            # paths, split dates, forbidden (leaky) columns
│   ├── load_data.py         # read tables with correct types + profile (rows, dtypes, missing)
│   ├── clean.py             # key / integrity / chronology checks + documented cleaning decisions
│   ├── build_dataset.py     # aggregate items & payments to order level, merge, target, features
│   ├── features.py          # feature list + preprocessing pipeline (fitted on train only)
│   ├── prepare.py           # step 1: raw -> processed data + audit tables
│   ├── eda.py               # step 2: six analysis questions, figures 01–10
│   ├── model.py             # step 3: baseline + 3 models, temporal split, evaluation, error analysis
│   └── make_docs.py         # writes docs/feature_dictionary.md
├── tests/test_pipeline.py   # keys, no duplicated orders, totals, target cases, leakage
├── docs/                    # ER diagram source, feature dictionary
└── results/
    ├── figures/             # 00–14 png
    ├── tables/              # audit, cleaning decisions, merge log, EDA tables, model results
    ├── experiments_log.csv  # every model setting tried with its validation score
    └── best_model.json
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset from the Kaggle link above (or use the course pack) and copy the 7 CSV files
listed above into `data/raw/`.

## Run

```bash
./run_all.sh            # uses `python`; or: PY=.venv/bin/python ./run_all.sh
```

or step by step:

```bash
cd src
python prepare.py       # audit + cleaning + order-level table
python eda.py           # figures 01-10 and EDA tables
python model.py         # models, figures 11-14, results/tables/model_results.csv
cd .. && python -m pytest -q tests
```

The full run takes about 1.5 minutes on a laptop. All random parts use seed 42, the split is
by date, so running it again gives the same tables and figures.

## Main decisions

* **Unit = one order.** Items and payments are aggregated to one row per order *before* being
  merged; the merge log (`results/tables/merge_log.csv`) shows the row count never changes.
* **Target `is_late`:** 1 if the delivery **day** is after the estimated **day**, 0 if on or
  before it. Only delivered orders with valid dates get a target; canceled, unavailable or
  still-shipping orders are *not* counted as on time.
* **Prediction moment = approval.** Inputs never include `order_status`,
  `order_delivered_customer_date`, `order_delivered_carrier_date`, actual delivery time or delay.
  Seller history uses only orders delivered *before* the approval time. Imputation, scaling and
  encoding are fitted on the training split only. `shipping_limit_date` is not used.
* **Temporal split** on `order_approved_at`: train < 2018-01-01 ≤ validation < 2018-05-01 ≤ test
  < 2018-09-01. Model settings and the decision threshold are chosen on validation; test is used
  once.

## Results (test period May–Aug 2018, late rate 4.4%)

| Model | Accuracy | Precision | Recall | F1 | Avg. precision |
|---|---|---|---|---|---|
| Baseline (always on time) | 0.956 | 0.000 | 0.000 | 0.000 | 0.044 |
| Logistic Regression (chosen on validation) | 0.549 | 0.066 | 0.703 | 0.121 | 0.071 |
| Random Forest | 0.750 | 0.066 | 0.354 | 0.111 | 0.064 |
| Gradient Boosting | 0.751 | 0.088 | 0.495 | 0.149 | 0.085 |

The models rank orders better than chance (late rate rises from 0.5% in the lowest score decile to
~8% in the highest) but the late rate changes a lot over time (5.6% train, 10.8% validation, 4.4%
test) and the estimated delivery times became shorter in mid-2018, so a threshold chosen on
validation flags far too many test orders. The baseline has the best accuracy, which shows why
accuracy alone is not useful here. Details and the full discussion are in the report.

## Team and contributions

| Responsibility | Member | Reviewer |
|---|---|---|
| Data quality and cleaning decisions | Mohammed Zaid Al-Nahhas (mhd_zaid_211096) | |
| Joins and features | | |
| Analysis and figures | | |
| Models and evaluation | | |
| Integration, tests and documentation | | |
