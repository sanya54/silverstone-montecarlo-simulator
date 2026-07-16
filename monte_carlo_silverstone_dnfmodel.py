"""
Monte Carlo F1 Race Strategy Simulator -- LAP-1-WEIGHTED DNF variant (2026 British GP)
==========================================================================================
Companion to monte_carlo_race_simulator.py. Same real grid, same qualifying data,
same tyre degradation and pit-loss logic -- the only thing that changes is how
retirements are modelled.

THE PROBLEM WITH THE ORIGINAL MODEL
--------------------------------------
The original script gives every driver the same flat 4.5% DNF chance, and if it
fires, picks a uniformly random lap anywhere in the race for the retirement. Real
incident risk isn't like that on either count: first-lap incidents are
disproportionately common (more cars bunched together immediately after the
start, less run-off margin used, first-corner pileups), and that risk is higher
the further back you start, because there are simply more cars around you in a
smaller space at the first braking zones. A flat, position-independent,
uniformly-timed DNF model misses both of those real patterns.

WHAT THIS SCRIPT DOES DIFFERENTLY
-------------------------------------
Retirement risk is split into two independent components instead of one flat
number:

  1. LAP-1 INCIDENT RISK -- scales with grid position:
         risk = LAP1_BASE_RISK * (1 + LAP1_GRID_SCALING * (grid_position - 1))
     A pole-sitter carries the base risk (1%); the last car on the grid carries
     roughly 4x that (~4.15%), reflecting the real pattern that first-lap chaos
     collects backmarkers disproportionately. These specific numbers are a
     documented modelling choice, not fitted from data -- isolating "which DNFs
     were caused by lap-1 incidents, by grid slot" wasn't feasible from the
     OpenF1 data pulled for this project.
  2. MECHANICAL DNF RISK -- a smaller, flat, position-independent probability
     (2.5%) applied for the rest of the race, representing genuine reliability
     failures that have nothing to do with where you started. If this fires,
     the retirement lap is drawn uniformly across the remaining race distance,
     same approach as the original model but now only covering genuine
     mechanical risk rather than everything.

A driver's overall DNF chance is whichever of these two fires first
(chronologically, a lap-1 incident would happen before a lap-30 mechanical
failure could). Averaged across the whole grid this comes out close to the
original model's flat 4.5% (so total attrition is comparable), but it's no
longer spread evenly -- it's now concentrated at the start and skewed toward the
back of the grid, both of which match how F1 retirements actually happen.

WHY THIS MATTERS FOR THE SIMULATION'S BEHAVIOUR
----------------------------------------------------
Front-runners barely change (their lap-1 risk is close to the original flat
number anyway). What changes is the tail of the grid: backmarkers now carry
real extra jeopardy that the original model didn't give them, which matters
for points-probability estimates in the midfield and for anyone building a
grid-position-dependent view of risk (e.g. a team deciding whether a
conservative or aggressive opening-lap approach is worth it from a slot further
back).

OUTPUTS
-------
  - race_probabilities_dnfmodel.csv
  - simulation_results_dnfmodel.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. SAME REAL GRID AS THE ORIGINAL SCRIPT -- 2026 British GP, Silverstone
# ---------------------------------------------------------------------------
drivers = [
    ("Antonelli",   "Mercedes",     1,  88.111),
    ("Leclerc",     "Ferrari",      2,  88.286),
    ("Hamilton",    "Ferrari",      3,  88.458),
    ("Russell",     "Mercedes",     4,  88.481),
    ("Hadjar",      "Red Bull",     5,  88.746),
    ("Norris",      "McLaren",      6,  88.877),
    ("Verstappen",  "Red Bull",     7,  88.893),
    ("Piastri",     "McLaren",      8,  89.032),
    ("Lindblad",    "Racing Bulls", 9,  89.305),
    ("Lawson",      "Racing Bulls", 10, 89.716),
    ("Bortoleto",   "Audi",         11, 89.461),
    ("Hulkenberg",  "Audi",         12, 90.076),
    ("Bearman",     "Haas",         13, 90.501),
    ("Sainz",       "Williams",     14, 90.623),
    ("Gasly",       "Alpine",       15, 90.063),
    ("Albon",       "Williams",     16, 90.743),
    ("Ocon",        "Haas",         17, 90.680),
    ("Bottas",      "Cadillac",     18, 91.227),
    ("Colapinto",   "Alpine",       19, 91.321),
    ("Perez",       "Cadillac",     20, 91.940),
    ("Alonso",      "Aston Martin", 21, 93.025),
    ("Stroll",      "Aston Martin", 22, 92.863),
]

actual_finish = {
    "Leclerc": 1, "Russell": 2, "Hamilton": 3, "Norris": 4, "Hadjar": 5,
    "Lawson": 6, "Lindblad": 7, "Bortoleto": 8, "Colapinto": 9, "Gasly": 10,
    "Piastri": 11, "Sainz": 12, "Bearman": 13, "Ocon": 14, "Perez": 15,
    "Antonelli": 16, "Bottas": 17, "Alonso": 18, "Stroll": 19,
    "Verstappen": "DNF", "Albon": "DNF", "Hulkenberg": "DNF",
}

names = [d[0] for d in drivers]
teams = [d[1] for d in drivers]
grid = np.array([d[2] for d in drivers])
quali = np.array([d[3] for d in drivers])
n = len(drivers)

pole = quali.min()
quali_gap = quali - pole

# ---------------------------------------------------------------------------
# 2. SAME BASE CONSTANTS AS THE ORIGINAL MODEL
# ---------------------------------------------------------------------------
QUALI_TO_RACE_SCALE = 0.45
base_pace_gap = quali_gap * QUALI_TO_RACE_SCALE

RACE_LAPS = 52
PIT_LOSS_GREEN = 21.0
PIT_LOSS_SC = 7.0
MEDIUM_DEG = 0.012
HARD_DEG = 0.010
LAP_NOISE_SD = 0.30

# --- NEW: lap-1-weighted, grid-dependent DNF model (replaces flat DNF_PROB) ---
LAP1_BASE_RISK = 0.010        # first-lap incident risk for the pole-sitter
LAP1_GRID_SCALING = 0.15      # +15% of the base risk per grid slot back from P1
                               # (P22 carries ~4.15x the pole-sitter's lap-1 risk)
MECHANICAL_DNF_PROB = 0.025   # flat, position-independent risk for the rest of
                               # the race (genuine reliability failures)

lap1_risk = LAP1_BASE_RISK * (1 + LAP1_GRID_SCALING * (grid - 1))
print("Lap-1 incident risk by grid position (vs. original model's flat 4.5% DNF, any lap):")
for nm, gp, risk in zip(names, grid, lap1_risk):
    print(f"  P{gp:>2} {nm:<11} lap-1 risk: {risk*100:.2f}%  (+ {MECHANICAL_DNF_PROB*100:.1f}% flat mechanical risk)")
print()

N_RUNS = 20000

# ---------------------------------------------------------------------------
# 3. MONTE CARLO SIMULATION -- identical strategy/SC/degradation logic to the
#    original, only the DNF section changes
# ---------------------------------------------------------------------------
finish_pos = np.zeros((N_RUNS, n), dtype=int)
stop_count = np.zeros((N_RUNS, n), dtype=int)
dnf_lap1_count = np.zeros(n, dtype=int)
dnf_mechanical_count = np.zeros(n, dtype=int)

for i in range(N_RUNS):
    n_sc = rng.choice([0, 1, 2, 3], p=[0.10, 0.25, 0.40, 0.25])
    sc_laps = np.sort(rng.choice(np.arange(10, 48), size=n_sc, replace=False)) if n_sc else np.array([])

    planned_pit = rng.integers(23, 33, size=n)
    pitted_under_sc = np.zeros(n, dtype=bool)
    pit1_lap = planned_pit.copy()
    if n_sc >= 1:
        sc1 = sc_laps[0]
        eligible = (sc1 >= 12) & (np.abs(planned_pit - sc1) <= 7)
        react = eligible & (rng.random(n) < 0.85)
        pit1_lap = np.where(react, sc1, planned_pit)
        pitted_under_sc = react

    pit_loss = np.where(pitted_under_sc, PIT_LOSS_SC, PIT_LOSS_GREEN)
    stops = np.ones(n, dtype=int)
    pit2_lap = np.full(n, -1)

    if n_sc >= 2:
        sc2 = sc_laps[1]
        eligible2 = pitted_under_sc & ((sc2 - pit1_lap) >= 8)
        react2 = eligible2 & (rng.random(n) < 0.75)
        pit_loss = np.where(react2, pit_loss + PIT_LOSS_SC, pit_loss)
        stops = np.where(react2, 2, stops)
        pit2_lap = np.where(react2, sc2, pit2_lap)

    stint1 = pit1_lap
    two_stop = pit2_lap >= 0
    stint2 = np.where(two_stop, pit2_lap - pit1_lap, RACE_LAPS - pit1_lap)
    stint3 = np.where(two_stop, RACE_LAPS - pit2_lap, 0)
    deg = (MEDIUM_DEG * stint1 * (stint1 + 1) / 2
           + HARD_DEG * stint2 * (stint2 + 1) / 2
           + HARD_DEG * stint3 * (stint3 + 1) / 2)

    noise = rng.normal(0, LAP_NOISE_SD * np.sqrt(RACE_LAPS), size=n)
    total_time = base_pace_gap * RACE_LAPS + deg + pit_loss + noise

    # --- Lap-1-weighted, grid-dependent DNF model ---
    lap1_incident = rng.random(n) < lap1_risk
    mechanical_incident = (~lap1_incident) & (rng.random(n) < MECHANICAL_DNF_PROB)
    dnf = lap1_incident | mechanical_incident

    retirement_lap = np.where(
        lap1_incident, 1,
        np.where(mechanical_incident, rng.integers(2, RACE_LAPS, size=n), RACE_LAPS)
    )
    dnf_lap1_count += lap1_incident.astype(int)
    dnf_mechanical_count += mechanical_incident.astype(int)

    sort_key = np.where(dnf, 1_000_000 - retirement_lap, total_time)

    order = np.argsort(sort_key)
    finish_pos[i, order] = np.arange(1, n + 1)
    stop_count[i, :] = stops

# ---------------------------------------------------------------------------
# 4. AGGREGATE RESULTS
# ---------------------------------------------------------------------------
win_prob = (finish_pos == 1).mean(axis=0)
podium_prob = (finish_pos <= 3).mean(axis=0)
points_prob = (finish_pos <= 10).mean(axis=0)
avg_finish = finish_pos.mean(axis=0)
avg_stops = stop_count.mean(axis=0)
dnf_rate = ((dnf_lap1_count + dnf_mechanical_count) / N_RUNS * 100)

results = pd.DataFrame({
    "driver": names, "team": teams, "grid": grid,
    "win_prob_%": (win_prob * 100).round(2),
    "podium_prob_%": (podium_prob * 100).round(2),
    "points_prob_%": (points_prob * 100).round(2),
    "avg_sim_finish": avg_finish.round(2),
    "avg_stops": avg_stops.round(2),
    "dnf_rate_%": dnf_rate.round(2),
    "actual_finish": [actual_finish[nm] for nm in names],
}).sort_values("win_prob_%", ascending=False).reset_index(drop=True)

results.to_csv("race_probabilities_dnfmodel.csv", index=False)
print(results.to_string(index=False))

actual_numeric = results["actual_finish"].apply(lambda v: v if isinstance(v, (int, float)) else n + 1)
spearman = results["avg_sim_finish"].corr(actual_numeric, method="spearman")
print(f"\nSpearman correlation (pre-race expected finish vs actual finish): {spearman:.3f}")

overall_dnf_rate = (dnf_lap1_count + dnf_mechanical_count).sum() / (N_RUNS * n) * 100
print(f"Overall average DNF rate across the field: {overall_dnf_rate:.2f}% "
      f"(original model's flat rate: 4.50%)")

front_half = results[results["grid"] <= 11]["dnf_rate_%"].mean()
back_half = results[results["grid"] > 11]["dnf_rate_%"].mean()
print(f"DNF rate, front half of grid (P1-11): {front_half:.2f}%  |  back half (P12-22): {back_half:.2f}%")

# ---------------------------------------------------------------------------
# 5. CHARTS
# ---------------------------------------------------------------------------
top10 = results.head(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
x = np.arange(len(top10))
width = 0.25
ax.bar(x - width, top10["win_prob_%"], width, label="Win %")
ax.bar(x, top10["podium_prob_%"], width, label="Podium %")
ax.bar(x + width, top10["points_prob_%"], width, label="Points %")
ax.set_xticks(x)
ax.set_xticklabels(top10["driver"], rotation=45, ha="right")
ax.set_ylabel("Probability (%)")
ax.set_title("Lap-1-weighted DNF model\n2026 British GP (20,000 simulations)")
ax.legend()

ax = axes[1]
by_grid = results.sort_values("grid")
ax.bar(by_grid["grid"], by_grid["dnf_rate_%"], color="#d62728")
ax.axhline(4.5, color="gray", linestyle="--", linewidth=1, label="original model (flat 4.5%)")
ax.set_xlabel("Grid position")
ax.set_ylabel("DNF rate (%)")
ax.set_title("DNF rate by grid slot: this model vs. the original's flat rate")
ax.legend()

plt.tight_layout()
plt.savefig("simulation_results_dnfmodel.png", dpi=150)
print("\nSaved race_probabilities_dnfmodel.csv and simulation_results_dnfmodel.png")
