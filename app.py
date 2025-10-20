# The Global Classifier Challenge — Streamlit App (v1, multi-user friendly)
# Updated: prefill fix, robust charts, team name in outputs, combined downloads, forms, perf tweaks
# NEW: training in earlier rounds unlocks L1/L2 capabilities in later rounds

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Global Classifier Challenge", layout="wide")

# -----------------------------
# Base data & configuration
# -----------------------------
EVENTS_BASE = [
    {"event":"Sydney Oceania Open","city":"Sydney","needs":{"L1":2,"L2":1}},
    {"event":"Tokyo Para Games","city":"Tokyo","needs":{"L1":1,"L2":2}},
    {"event":"Delhi Invitational","city":"Delhi","needs":{"L1":2,"L2":1}},
    {"event":"Nairobi Championship","city":"Nairobi","needs":{"L1":2,"L2":0}},
    {"event":"Cairo Continental","city":"Cairo","needs":{"L1":2,"L2":1}},
    {"event":"Paris Grand Prix","city":"Paris","needs":{"L1":1,"L2":2}},
    {"event":"São Paulo Classic","city":"São Paulo","needs":{"L1":2,"L2":0}},
    {"event":"Toronto Open","city":"Toronto","needs":{"L1":2,"L2":1}},
    {"event":"Cape Town Masters","city":"Cape Town","needs":{"L1":1,"L2":2}},
    {"event":"Auckland Para Cup","city":"Auckland","needs":{"L1":2,"L2":0}},
]

CLASSIFIERS = [
    {"id":"C-1","base":"Madrid","level":"L2","cost_per_km":0.22},
    {"id":"C-2","base":"Warsaw","level":"L1","cost_per_km":0.19},
    {"id":"C-3","base":"Seoul","level":"L1","cost_per_km":0.20},
    {"id":"C-4","base":"Buenos Aires","level":"L2","cost_per_km":0.23},
    {"id":"C-5","base":"Vancouver","level":"L1","cost_per_km":0.20},
    {"id":"C-6","base":"Lagos","level":"L2","cost_per_km":0.25},
    {"id":"C-7","base":"London","level":"L2","cost_per_km":0.21},
    {"id":"C-8","base":"Bangkok","level":"L1","cost_per_km":0.18},
    {"id":"C-9","base":"Lima","level":"L1","cost_per_km":0.20},
    {"id":"C-10","base":"Rome","level":"L2","cost_per_km":0.23},
    {"id":"C-11","base":"Johannesburg","level":"L1","cost_per_km":0.19},
    {"id":"C-12","base":"Kuala Lumpur","level":"L1","cost_per_km":0.20},
    {"id":"C-13","base":"Santiago","level":"L2","cost_per_km":0.24},
    {"id":"C-14","base":"Brisbane","level":"L1","cost_per_km":0.19},
    {"id":"C-15","base":"Istanbul","level":"L2","cost_per_km":0.23},
    {"id":"C-16","base":"Mumbai","level":"L1","cost_per_km":0.18},
    {"id":"C-17","base":"Nairobi","level":"L1","cost_per_km":0.20},
    {"id":"C-18","base":"Helsinki","level":"L2","cost_per_km":0.21},
]

COORDS = {
    "Sydney":(-33.8688,151.2093), "Tokyo":(35.6762,139.6503), "Delhi":(28.6139,77.2090),
    "Nairobi":(-1.2864,36.8172), "Cairo":(30.0444,31.2357), "Paris":(48.8566,2.3522),
    "São Paulo":(-23.5505,-46.6333), "Toronto":(43.6532,-79.3832), "Cape Town":(-33.9249,18.4241),
    "Auckland":(-36.8485,174.7633),
    "Madrid":(40.4168,-3.7038), "Warsaw":(52.2297,21.0122), "Seoul":(37.5665,126.9780),
    "Buenos Aires":(-34.6037,-58.3816), "Vancouver":(49.2827,-123.1207), "Lagos":(6.5244,3.3792),
    "London":(51.5074,-0.1278), "Bangkok":(13.7563,100.5018), "Lima":(-12.0464,-77.0428),
    "Rome":(41.9028,12.4964), "Johannesburg":(-26.2041,28.0473), "Kuala Lumpur":(3.1390,101.6869),
    "Santiago":(-33.4489,-70.6693), "Brisbane":(-27.4698,153.0251), "Istanbul":(41.0082,28.9784),
    "Mumbai":(19.0760,72.8777), "Helsinki":(60.1699,24.9384),
    "Mexico City":(19.4326,-99.1332),
}

TRAIN_L1_COST, TRAIN_L2_COST, TRAINING_HUB_COST, REMOTE_EVENT_COST = 2000, 4000, 3000, 1500
CARBON_THRESHOLD_KM, CARBON_MULTIPLIER = 8000, 1.5

# -----------------------------
# Cache base data / helpers
# -----------------------------
@st.cache_data
def get_events_base() -> List[dict]:
    # return a deep copy to avoid mutation across sessions
    return [dict(event=e["event"], city=e["city"], needs=dict(e["needs"])) for e in EVENTS_BASE]

@st.cache_data
def get_classifiers_df() -> pd.DataFrame:
    return pd.DataFrame(CLASSIFIERS)

def haversine_km(a, b) -> float:
    R=6371.0
    lat1,lon1=map(math.radians,a); lat2,lon2=map(math.radians,b)
    dlat=lat2-lat1; dlon=lon2-lon1
    h=math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

@st.cache_data
def dist_km(base_city: str, event_city: str) -> int:
    if base_city not in COORDS or event_city not in COORDS:
        return 0
    return round(haversine_km(COORDS[base_city], COORDS[event_city]))

# Fast lookups
CLF_BY_ID = {c["id"]: c for c in CLASSIFIERS}
BASE_LEVELS: Dict[str, Set[str]] = {c["id"]: {c["level"]} for c in CLASSIFIERS}

@st.cache_data
def precompute_costs():
    """Precompute base->city cost (distance * €/km)."""
    costs = {}
    # include potential new event for robustness
    all_events = EVENTS_BASE + [{"event":"Mexico City Open","city":"Mexico City","needs":{"L1":2,"L2":0}}]
    for c in CLASSIFIERS:
        for e in all_events:
            city = e["city"]
            d = dist_km(c["base"], city)
            costs[(c["id"], city)] = d * c["cost_per_km"]
    return costs

BASE_COSTS = precompute_costs()

# -----------------------------
# Training upgrades across rounds
# -----------------------------
# We store cumulative upgrades after each completed round:
#   st.session_state.upgrades_after_R1: Dict[cid, Set["L1"|"L2"]]
#   st.session_state.upgrades_after_R2: Dict[cid, Set["L1"|"L2"]]
st.session_state.setdefault("upgrades_after_R1", {})
st.session_state.setdefault("upgrades_after_R2", {})

def compute_upgrades_from_assignments(assignments: Dict[str, dict]) -> Dict[str, Set[str]]:
    """
    Any classifier assigned to an event with 'Train L1' gains L1,
    with 'Train L2' gains L2 — for FUTURE rounds.
    """
    gains: Dict[str, Set[str]] = {}
    for ev_name, plan in assignments.items():
        training = plan.get("training", [])
        cids = plan.get("classifiers", [])
        add_l1 = "Train L1" in training
        add_l2 = "Train L2" in training
        if not (add_l1 or add_l2) or not cids:
            continue
        for cid in cids:
            if cid not in gains:
                gains[cid] = set()
            if add_l1:
                gains[cid].add("L1")
            if add_l2:
                gains[cid].add("L2")
    return gains

def get_effective_levels_for_round(round_key: str) -> Dict[str, Set[str]]:
    """
    Effective levels = base levels + upgrades from prior rounds only.
    - R1: base only
    - R2: base + upgrades_after_R1
    - R3: base + upgrades_after_R1 + upgrades_after_R2
    """
    eff = {cid: set(levels) for cid, levels in BASE_LEVELS.items()}
    if round_key in ("R2", "R3"):
        for cid, adds in st.session_state.get("upgrades_after_R1", {}).items():
            eff.setdefault(cid, set()).update(adds)
    if round_key == "R3":
        for cid, adds in st.session_state.get("upgrades_after_R2", {}).items():
            eff.setdefault(cid, set()).update(adds)
    return eff

def label_for_classifier(cid: str, eff_levels: Dict[str, Set[str]]) -> str:
    """Show base level and arrows for upgrades, e.g., 'C-2 • Warsaw • L1 (↑L2)'."""
    base_level = next(iter(BASE_LEVELS[cid]))
    eff = eff_levels.get(cid, {base_level})
    gained = sorted(list(eff - {base_level}))
    clf = CLF_BY_ID[cid]
    if gained:
        return f"{cid} • {clf['base']} • {base_level} (↑{','.join(gained)}) • €{clf['cost_per_km']}/km"
    else:
        return f"{cid} • {clf['base']} • {base_level} • €{clf['cost_per_km']}/km"

# -----------------------------
# Core compute
# -----------------------------
@dataclass
class Scenario:
    travel_mult: float = 1.0
    carbon: bool = False
    visa_blocked: Tuple[str, ...] = ()
    free_remote_events: int = 0
    free_train_L1: int = 0
    new_event: bool = False

def summarize_round(events: List[dict], assignments: Dict[str, dict], scen: Scenario,
                    eff_levels: Dict[str, Set[str]]) -> Tuple[pd.DataFrame, float, bool, Dict[str, float]]:
    rows=[]; total=0.0; coverage_ok=True
    sums = {"travel":0.0,"training":0.0,"remote":0.0}
    scen_local = Scenario(**scen.__dict__)

    for ev in events:
        ev_name, city = ev["event"], ev["city"]
        needL1, needL2 = ev["needs"]["L1"], ev["needs"]["L2"]
        plan = assignments.get(ev_name, {"classifiers":[], "training":[], "remote":False})
        cids = plan.get("classifiers", []); training = plan.get("training", []); remote = plan.get("remote", False)

        # Free remote applied lazily once
        if scen_local.free_remote_events > 0 and not remote:
            remote = True
            scen_local.free_remote_events -= 1

        haveL1=0; haveL2=0; travel_total=0.0
        for cid in cids:
            if cid in scen_local.visa_blocked:
                continue
            clf = CLF_BY_ID.get(cid)
            if not clf:
                continue

            # coverage using effective levels
            levels = eff_levels.get(cid, {clf["level"]})
            if "L1" in levels: haveL1 += 1
            if "L2" in levels: haveL2 += 1

            base_cost = BASE_COSTS.get((cid, city), 0.0) * scen_local.travel_mult
            if scen_local.carbon:
                d = dist_km(clf["base"], city)
                if d >= CARBON_THRESHOLD_KM:
                    base_cost *= CARBON_MULTIPLIER
            travel_total += base_cost

        covered = (haveL1>=needL1) and (haveL2>=needL2)
        coverage_ok = coverage_ok and covered
        training_total = sum(
            TRAIN_L1_COST if t=="Train L1" else TRAIN_L2_COST if t=="Train L2" else TRAINING_HUB_COST
            for t in training
        )
        remote_total = REMOTE_EVENT_COST if remote else 0.0
        event_total = travel_total + training_total + remote_total
        total += event_total
        sums["travel"] += travel_total; sums["training"] += training_total; sums["remote"] += remote_total

        rows.append([ev_name, covered, haveL1, haveL2, round(travel_total,2), training_total, remote_total, round(event_total,2)])

    # Scenario free L1 trainings (budget) applied cross-event
    if scen_local.free_train_L1:
        l1_trains = sum(1 for p in assignments.values() for t in p.get("training",[]) if t=="Train L1")
        deduct = min(l1_trains, scen_local.free_train_L1) * TRAIN_L1_COST
        total = max(0.0, total - deduct)
        sums["training"] = max(0.0, sums["training"] - deduct)

    df = pd.DataFrame(rows, columns=["Event","Covered","Have L1","Have L2","Travel €","Training €","Remote €","Event Total €"])
    df.insert(0, "Team", st.session_state.get("team_name", "Unknown"))
    return df, round(total,2), coverage_ok, {k: round(v,2) for k,v in sums.items()}

# -----------------------------
# Sidebar (team, nav)
# -----------------------------
st.sidebar.title("Global Classifier Challenge")
team_name = st.sidebar.text_input("Team name", value=st.session_state.get("team_name", "Team A"))
st.session_state["team_name"] = team_name

st.sidebar.markdown("### Navigation")
tabs = st.sidebar.radio(
    "Go to",
    ["Overview", "Data", "Round 1", "Scenarios", "Round 2", "Round 3", "Results", "Routes Map"],
    index=0,
)

# key-version trick for Prefill so widgets rebuild with new defaults
st.session_state.setdefault("keyver_R1", 0)
st.session_state.setdefault("keyver_R2", 0)
st.session_state.setdefault("keyver_R3", 0)

# Keep a per-session copy of events
if "events" not in st.session_state:
    st.session_state.events = get_events_base()

# -----------------------------
# Overview
# -----------------------------
if tabs == "Overview":
    st.title("🧩 Global Classifier Challenge")
    st.write(
        """
        **Goal:** Cover all events while minimizing cost across Years 1–3.
        - **Round 1:** baseline plan — assign classifiers, training, remote.
        - **Round 2:** apply scenarios (Travel +20%, Carbon penalty, Visa delays, Free remote, Free L1 budget), adapt plan.
        - **Round 3:** optional stabilization year (supports a **new event** scenario).
        - **Results:** charts and CSV downloads.
        """
    )
    st.info("Tip: Each user sees only their own session. Add a Team name so exported files include your team code.")

# -----------------------------
# Data tab (tables + map)
# -----------------------------
elif tabs == "Data":
    st.header("Data")
    events_df = pd.DataFrame([
        {"Event": e["event"], "City": e["city"], "Needs L1": e["needs"]["L1"], "Needs L2": e["needs"]["L2"]}
        for e in st.session_state.events
    ])
    st.subheader("Events")
    st.dataframe(events_df, use_container_width=True)

    st.subheader("Classifiers")
    clf_df = get_classifiers_df()[["id","base","level","cost_per_km"]].rename(columns={"id":"ID","base":"Base","level":"Level","cost_per_km":"€/km"})
    st.dataframe(clf_df, use_container_width=True)

    with st.expander("Map: Events (red) & Classifier Bases (blue)", expanded=False):
        m = folium.Map(location=[15,10], zoom_start=2)
        # events
        for e in st.session_state.events:
            lat,lon = COORDS[e["city"]]
            folium.Marker([lat,lon], popup=e["event"], tooltip=f"Event: {e['event']}", icon=folium.Icon(color="red", icon="flag")).add_to(m)
        # bases (unique)
        seen=set()
        for c in CLASSIFIERS:
            if c["base"] in seen: continue
            seen.add(c["base"])
            lat,lon = COORDS[c["base"]]
            folium.Marker([lat,lon], popup=c["base"], tooltip=f"Base: {c['base']}", icon=folium.Icon(color="blue")).add_to(m)
        st_folium(m, width=1000, height=520)

# -----------------------------
# UI builders for rounds
# -----------------------------
def round_planner_ui(round_key: str):
    """Return assignments dict for the given round, using effective levels (with upgrades)."""
    ver = st.session_state.get(f"keyver_{round_key}", 0)
    eff_levels = get_effective_levels_for_round(round_key)
    st.caption(f"Select classifiers, training, and remote for **{round_key}**")

    prefill = st.session_state.get(f"assign_{round_key}", {})
    assignments = {}

    # Build lists from effective levels
    L1_ids = [cid for cid, levels in eff_levels.items() if "L1" in levels]
    L2_ids = [cid for cid, levels in eff_levels.items() if "L2" in levels]

    for ev in st.session_state.events:
        ev_name = ev["event"]
        with st.expander(f"{ev_name} — needs L1:{ev['needs']['L1']}  L2:{ev['needs']['L2']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Select L1**")
                l1_selected = []
                for cid in L1_ids:
                    key = f"{round_key}_v{ver}_{ev_name}_L1_{cid}"
                    default = cid in prefill.get(ev_name, {}).get("classifiers", [])
                    on = st.checkbox(label_for_classifier(cid, eff_levels), value=default, key=key)
                    if on: l1_selected.append(cid)
            with col2:
                st.markdown("**Select L2**")
                l2_selected = []
                for cid in L2_ids:
                    key = f"{round_key}_v{ver}_{ev_name}_L2_{cid}"
                    default = cid in prefill.get(ev_name, {}).get("classifiers", [])
                    on = st.checkbox(label_for_classifier(cid, eff_levels), value=default, key=key)
                    if on: l2_selected.append(cid)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                t1 = st.checkbox(f"Train L1 (+€{TRAIN_L1_COST})",
                                 value=("Train L1" in prefill.get(ev_name, {}).get("training", [])),
                                 key=f"{round_key}_v{ver}_{ev_name}_trainL1")
            with c2:
                t2 = st.checkbox(f"Train L2 (+€{TRAIN_L2_COST})",
                                 value=("Train L2" in prefill.get(ev_name, {}).get("training", [])),
                                 key=f"{round_key}_v{ver}_{ev_name}_trainL2")
            with c3:
                hub = st.checkbox(f"Training Hub (+€{TRAINING_HUB_COST})",
                                  value=("Training Hub" in prefill.get(ev_name, {}).get("training", [])),
                                  key=f"{round_key}_v{ver}_{ev_name}_hub")
            with c4:
                remote = st.checkbox(f"Remote (+€{REMOTE_EVENT_COST})",
                                     value=bool(prefill.get(ev_name, {}).get("remote", False)),
                                     key=f"{round_key}_v{ver}_{ev_name}_remote")

            training = []
            if t1: training.append("Train L1")
            if t2: training.append("Train L2")
            if hub: training.append("Training Hub")

            assignments[ev_name] = {
                "classifiers": l1_selected + l2_selected,
                "training": training,
                "remote": remote,
            }

    # Save for later use
    st.session_state[f"assign_{round_key}"] = assignments
    return assignments, eff_levels

def scenarios_ui(label: str) -> Scenario:
    opts = [
        "Travel +20%",
        "Carbon penalty (+50% for trips ≥ 8,000 km)",
        "Visa delays (block C-1 & C-2)",
        "Free remote for 1 event",
        "Extra training budget (2 free L1)",
        "New event added (Mexico City, 2 L1)  [Round 3 only]",
    ]
    sel = st.multiselect(label, opts, default=[])
    scen = Scenario()
    if "Travel +20%" in sel: scen.travel_mult *= 1.2
    if "Carbon penalty (+50% for trips ≥ 8,000 km)" in sel: scen.carbon = True
    if "Visa delays (block C-1 & C-2)" in sel: scen.visa_blocked = ("C-1","C-2")
    if "Free remote for 1 event" in sel: scen.free_remote_events = 1
    if "Extra training budget (2 free L1)" in sel: scen.free_train_L1 = 2
    if "New event added (Mexico City, 2 L1)  [Round 3 only]" in sel:
        scen.new_event = True
    return scen

def show_results_tables_and_charts(df1=None, t1=None, mix1=None, df2=None, t2=None, mix2=None, df3=None, t3=None, mix3=None):
    # Totals chart
    if t1 is not None and t2 is not None:
        labels=["Year 1","Year 2"] + (["Year 3"] if t3 is not None else [])
        vals=[t1,t2] + ([t3] if t3 is not None else [])
        fig = plt.figure()
        plt.bar(labels, vals)
        plt.title("Total Cost by Year")
        plt.ylabel("€")
        st.pyplot(fig)

    # Per-event bars (robust to different event sets per year)
    if df1 is not None:
        def to_series(df, label):
            s = df.set_index("Event")["Event Total €"]
            s.name = label
            return s

        frames = [to_series(df1, "Y1")]
        if df2 is not None: frames.append(to_series(df2, "Y2"))
        if df3 is not None: frames.append(to_series(df3, "Y3"))

        merged = pd.concat(frames, axis=1).fillna(0.0)
        all_events = list(merged.index)
        x = range(len(all_events))
        n_series = merged.shape[1]
        width = 0.8 / n_series

        fig = plt.figure()
        for i, col in enumerate(merged.columns):
            positions = [xi + (i - (n_series - 1)/2) * width for xi in x]
            plt.bar(positions, merged[col].values.tolist(), width, label=col)

        plt.xticks(list(x), all_events, rotation=45, ha="right")
        plt.title("Per-Event Cost")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Cost mix pies (guard against zero totals)
    def pie(mix, title):
        mix = mix or {}
        vals = [
            float(mix.get("travel", 0) or 0),
            float(mix.get("training", 0) or 0),
            float(mix.get("remote", 0) or 0),
        ]
        total = sum(vals)
        if total <= 0:
            st.info(f"{title}: no costs to plot yet.")
            return
        fig = plt.figure()
        plt.pie(vals, labels=["Travel","Training","Remote"], autopct="%1.0f%%", startangle=90)
        plt.title(title); plt.axis("equal")
        st.pyplot(fig)

    if mix1: pie(mix1, "Cost Mix – Year 1")
    if mix2: pie(mix2, "Cost Mix – Year 2")
    if mix3: pie(mix3, "Cost Mix – Year 3")

# -----------------------------
# Round 1
# -----------------------------
if tabs == "Round 1":
    st.header("Round 1 — Planning Year")

    with st.form("r1_form"):
        assignments, eff_levels = round_planner_ui("R1")
        submitted = st.form_submit_button("Calculate Round 1")
    if submitted:
        df1, t1, ok1, mix1 = summarize_round(st.session_state.events, assignments, Scenario(), eff_levels)
        st.session_state.df1, st.session_state.t1, st.session_state.ok1, st.session_state.mix1 = df1, t1, ok1, mix1

        # compute & store upgrades for use in R2
        st.session_state.upgrades_after_R1 = compute_upgrades_from_assignments(assignments)

        st.dataframe(df1, use_container_width=True)
        st.success(f"TOTAL (Year 1): €{t1:,.0f} | Coverage OK: {ok1}")
        st.download_button(
            "Download Round 1 CSV",
            df1.to_csv(index=False).encode("utf-8"),
            file_name=f"round1_results_{team_name}.csv",
            mime="text/csv",
        )

# -----------------------------
# Scenarios
# -----------------------------
elif tabs == "Scenarios":
    st.header("Scenarios")
    st.write("Pick scenarios for Round 2 and Round 3.")
    st.session_state.scen2 = scenarios_ui("Round 2 Scenarios")
    st.session_state.scen3 = scenarios_ui("Round 3 Scenarios (Year 3)")

    # apply new event if chosen for R3
    if st.session_state.scen3.new_event and not any(e["city"]=="Mexico City" for e in st.session_state.events):
        st.session_state.events.append({"event":"Mexico City Open","city":"Mexico City","needs":{"L1":2,"L2":0}})
        st.info("Added **Mexico City Open** (needs 2 L1).")

# -----------------------------
# Round 2
# -----------------------------
elif tabs == "Round 2":
    st.header("Round 2 — Disruption Year")
    can_prefill = bool(st.session_state.get("assign_R1"))
    if st.button("Prefill from Round 1", disabled=not can_prefill):
        st.session_state["assign_R2"] = st.session_state.get("assign_R1", {})
        st.session_state["keyver_R2"] += 1  # force widgets to rebuild with defaults
        st.success("Prefilled Round 2 from Round 1.")

    scen2 = st.session_state.get("scen2", Scenario())  # defaults if not set
    st.caption(f"Active Round 2 scenarios: {scen2.__dict__}")

    with st.form("r2_form"):
        assignments, eff_levels = round_planner_ui("R2")
        submitted = st.form_submit_button("Calculate Round 2")
    if submitted:
        df2, t2, ok2, mix2 = summarize_round(st.session_state.events, assignments, scen2, eff_levels)
        st.session_state.df2, st.session_state.t2, st.session_state.ok2, st.session_state.mix2 = df2, t2, ok2, mix2

        # compute & store upgrades for use in R3
        st.session_state.upgrades_after_R2 = compute_upgrades_from_assignments(assignments)

        st.dataframe(df2, use_container_width=True)
        st.success(f"TOTAL (Year 2): €{t2:,.0f} | Coverage OK: {ok2}")
        st.download_button(
            "Download Round 2 CSV",
            df2.to_csv(index=False).encode("utf-8"),
            file_name=f"round2_results_{team_name}.csv",
            mime="text/csv",
        )

# -----------------------------
# Round 3
# -----------------------------
elif tabs == "Round 3":
    st.header("Round 3 — Stabilization Year")
    can_prefill = bool(st.session_state.get("assign_R2"))
    if st.button("Prefill from Round 2", disabled=not can_prefill):
        st.session_state["assign_R3"] = st.session_state.get("assign_R2", {})
        st.session_state["keyver_R3"] += 1
        st.success("Prefilled Round 3 from Round 2.")

    scen3 = st.session_state.get("scen3", Scenario())
    st.caption(f"Active Round 3 scenarios: {scen3.__dict__}")

    with st.form("r3_form"):
        assignments, eff_levels = round_planner_ui("R3")
        submitted = st.form_submit_button("Calculate Round 3")
    if submitted:
        df3, t3, ok3, mix3 = summarize_round(st.session_state.events, assignments, scen3, eff_levels)
        st.session_state.df3, st.session_state.t3, st.session_state.ok3, st.session_state.mix3 = df3, t3, ok3, mix3

        st.dataframe(df3, use_container_width=True)
        st.success(f"TOTAL (Year 3): €{t3:,.0f} | Coverage OK: {ok3}")
        st.download_button(
            "Download Round 3 CSV",
            df3.to_csv(index=False).encode("utf-8"),
            file_name=f"round3_results_{team_name}.csv",
            mime="text/csv",
        )

# -----------------------------
# Results
# -----------------------------
elif tabs == "Results":
    st.header("Results & Charts")
    df1, t1, mix1 = st.session_state.get("df1"), st.session_state.get("t1"), st.session_state.get("mix1")
    df2, t2, mix2 = st.session_state.get("df2"), st.session_state.get("t2"), st.session_state.get("mix2")
    df3, t3, mix3 = st.session_state.get("df3"), st.session_state.get("t3"), st.session_state.get("mix3")
    if not (df1 is not None and t1 is not None):
        st.warning("Please calculate at least Round 1 first.")
    else:
        show_results_tables_and_charts(df1, t1, mix1, df2, t2, mix2, df3, t3, mix3)

        # Combined downloads (ALL rounds)
        import io, zipfile

        dfs = []
        if df1 is not None: d1 = df1.copy(); d1["Round"] = "Round 1"; dfs.append(d1)
        if df2 is not None: d2 = df2.copy(); d2["Round"] = "Round 2"; dfs.append(d2)
        if df3 is not None: d3 = df3.copy(); d3["Round"] = "Round 3"; dfs.append(d3)

        if dfs:
            all_df = pd.concat(dfs, ignore_index=True)
            # single CSV
            csv_bytes = all_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download ALL Rounds (CSV)",
                csv_bytes,
                file_name=f"all_rounds_{team_name}.csv",
                mime="text/csv",
            )
            # ZIP with per-round CSVs
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                if df1 is not None: zf.writestr(f"round1_{team_name}.csv", df1.to_csv(index=False))
                if df2 is not None: zf.writestr(f"round2_{team_name}.csv", df2.to_csv(index=False))
                if df3 is not None: zf.writestr(f"round3_{team_name}.csv", df3.to_csv(index=False))
            st.download_button(
                "📦 Download ZIP (Rounds 1–3)",
                zip_buf.getvalue(),
                file_name=f"results_{team_name}.zip",
                mime="application/zip",
            )

# -----------------------------
# Routes Map
# -----------------------------
elif tabs == "Routes Map":
    st.header("Routes Map")
    plan_choice = st.selectbox("Select plan to draw routes", ["Round 1","Round 2","Round 3"])
    key = {"Round 1":"assign_R1", "Round 2":"assign_R2", "Round 3":"assign_R3"}[plan_choice]
    plan = st.session_state.get(key, {})
    if not plan:
        st.info(f"No assignments found for {plan_choice}. Build and calculate that round first.")
    else:
        m = folium.Map(location=[15,10], zoom_start=2)
        # markers
        for e in st.session_state.events:
            lat,lon = COORDS[e["city"]]
            folium.Marker([lat,lon], popup=e["event"], tooltip=f"Event: {e['event']}",
                          icon=folium.Icon(color="red", icon="flag")).add_to(m)
        seen=set()
        for c in CLASSIFIERS:
            if c["base"] in seen: continue
            seen.add(c["base"])
            lat,lon = COORDS[c["base"]]
            folium.Marker([lat,lon], popup=c["base"], tooltip=f"Base: {c['base']}",
                          icon=folium.Icon(color="blue")).add_to(m)
        # routes
        for ev in st.session_state.events:
            ev_name, city = ev["event"], ev["city"]
            ev_lat, ev_lon = COORDS[city]
            for cid in plan.get(ev_name, {}).get("classifiers", []):
                clf = CLF_BY_ID.get(cid)
                if not clf: continue
                base_lat, base_lon = COORDS[clf["base"]]
                folium.PolyLine([(base_lat, base_lon), (ev_lat, ev_lon)], weight=2, opacity=0.7).add_to(m)
        st_folium(m, width=1000, height=560)
