# The Global Classifier Challenge — Streamlit App (v1, multi‑user friendly)

Plan classifier coverage for a global calendar of events while keeping costs in check across three rounds (years) and multiple disruption scenarios. The app supports many simultaneous teams via Streamlit session state and caching.

developments idea: Tiago Russomanno
code: AI+ Author 

---

## ✨ Key Features

* **Team name input** — separate session state per browser session; export files include your team name.
* **Data tab** — interactive tables of Events & Classifiers **plus a Folium map** of event locations (red) and classifier bases (blue).
* **Round 1 / Round 2 / Round 3 planners** — BIG checkbox panels per event for selecting **L1/L2 classifiers**, training options, and remote coverage.
* **Scenarios (R2 & R3)** — apply travel inflation, carbon penalties, visa blocks, free remote, extra L1 training budget, and an optional new event.
* **Results** — totals chart, per‑event bar charts, cost‑mix pie charts, and CSV downloads for each round.
* **Routes Map** — draws base→event lines for the selected round’s plan.
* **Caching** — base data and distance computations cached to scale to ~20 concurrent users comfortably.

---

## 🧩 Game Objective

Cover all events with required **L1** and **L2** classifiers each year while **minimizing total cost**. Manage trade‑offs between travel, training, and remote coverage—then adapt to shocks in later rounds.

---

## 🔧 Quickstart

### 1) Requirements

* Python **3.9+**
* pip

### 2) Install

```bash
# from your project root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit pandas matplotlib folium streamlit-folium
```

### 3) Run locally

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually [http://localhost:8501](http://localhost:8501)).

> **File name:** If your script isn’t `app.py`, replace accordingly (e.g., `streamlit run global_classifier.py`).

---

## 🗂️ App Structure (high level)

* **Config**: sets page title and wide layout.
* **Base Data**: events, classifier roster, city coordinates, and cost constants.
* **Caching**: `@st.cache_data` wrappers for events, classifier DataFrame, and distance calculations.
* **Core Logic**: `Scenario` dataclass and `summarize_round()` which computes coverage, totals, and cost breakdowns.
* **UI**:

  * **Sidebar**: app title, team name input, and navigation radio.
  * **Overview** tab: goal and round descriptions.
  * **Data** tab: data tables and Folium map of events & bases.
  * **Round 1/2/3**: planner UIs with checkbox panels per event (L1/L2 picks, training, remote) and **Calculate** buttons.
  * **Scenarios** tab: pick Round 2 and 3 scenarios (inflation, carbon, visa, remote, training subsidy, new event).
  * **Results** tab: charts and CSV downloads.
  * **Routes Map** tab: base→event polylines for the selected round.

---

## 🧮 Cost Model (in code)

* **Travel cost** = great‑circle distance (km) × classifier‑specific €/km × scenario multiplier.
* **Carbon penalty**: +50% for individual trips ≥ 8,000 km (when enabled).
* **Training options** (per event):

  * Train L1: €2,000
  * Train L2: €4,000
  * Training Hub: €3,000
* **Remote coverage**: €1,500 per event (flat).
* **Scenario freebies**:

  * *Free remote for 1 event* (applied lazily to first eligible event).
  * *Extra training budget* = 2 free L1 trainings (deducted after per‑event totals).

All per‑event costs and totals are summarized in a table; yearly totals and cost mixes are visualized.

---

## 🧭 UI Walkthrough

### Overview

Explains the objective and the three rounds:

* **Round 1**: Baseline plan.
* **Round 2**: Disruption year. Apply scenarios and adapt the plan.
* **Round 3**: Stabilization year. Optionally add a new event.

### Data

* Two intera
