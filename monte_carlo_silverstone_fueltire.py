"""
Monte Carlo F1 Race Strategy Simulator -- FUEL/TIRE-SEPARATED variant (2026 British GP)
==========================================================================================
Companion to monte_carlo_race_simulator.py. Same real grid, same qualifying data,
same pit-loss and DNF logic -- the only thing that changes is how tyre degradation
is derived.

THE PROBLEM WITH THE ORIGINAL MODEL
--------------------------------------
The original script fit degradation directly from Leclerc's real lap times and got
a near-flat slope (+0.011s/lap on the Medium, -0.004s/lap on the Hard). That's the
NET effect of two things happening at once and pulling in opposite directions:
tyres wearing out (which slows the car down) and fuel burning off (which speeds
the car up, since a lighter car is a faster car). The real dataset can't separate
them -- it only measures the combined result. Using the net number is internally
consistent for predicting total race time, but it understates how much tyre
management actually matters, because a lot of real tyre wear is being masked by
fuel burn-off in that raw number.

WHAT THIS SCRIPT DOES DIFFERENTLY
-------------------------------------
It models the two effects separately and adds them back together explicitly:

  1. FUEL EFFECT -- a well-established, publicly documented rule of thumb in F1
     is that carrying an extra 10kg of fuel costs a car roughly 0.3s of lap time
     (i.e. ~0.03s/lap per kg). A 2026-spec car burns its ~110kg race fuel load
     over the race, so at Silverstone's 52 laps that's about 2.12kg/lap, or
     ~0.0635s/lap FASTER each successive lap purely from getting lighter. This
     number is NOT derived from the OpenF1 data pulled for this project (fuel
     load isn't in that dataset) -- it's a standard industry estimate, documented
     here rather than fitted.
  2. CORRECTED TYRE DEGRADATION -- since net_observed_slope = tyre_deg_rate -
     fuel_effect_rate, we can solve for the tyre-only rate:
         tyre_deg_rate = net_observed_slope + fuel_effect_rate
     Medium: 0.011 + 0.0635 ~= 0.075s/lap of tyre age (vs 0.012 in the original)
     Hard:  -0.004 + 0.0635 ~= 0.060s/lap of tyre age (vs 0.010 in the original)
     Both corrected rates are roughly 5-6x higher than the net numbers the
     original model used.

WHY THIS MATTERS FOR THE SIMULATION'S BEHAVIOUR
----------------------------------------------------
The fuel effect itself is applied identically to every driver (same fuel load,
same burn rate assumed for the whole field), so it's mathematically a wash for
finishing order -- it shifts everyone's total time by the same amount and changes
nothing about who wins. It's included below purely for transparency and so the
absolute total-time numbers mean something, but it is NOT what changes the
results.
What DOES change the results is using the much higher corrected tyre degradation
rate. Since real tyre wear turns out to be substantially worse than the
original's flat, near-zero number suggested, stint length now matters far more
in absolute terms -- a driver stuck out too long on old tyres pays a real,
meaningful price. Counterintuitively, this actually NARROWS the 1-stop-vs-2-stop
gap rather than widening it (18.4s in the original model vs ~4.7s here): because
degradation cost scales with the square of stint length, a 2-stop's three
shorter stints benefit proportionally more from higher wear rates than a
1-stop's two longer ones, which partly offsets the extra pit stop. The practical
read: tyre management is much higher-stakes in this version, and the strategic
call between one and two stops is closer than the original model suggested, not
more lopsided. See the printed strategy-comparison output below for the numbers.

OUTPUTS
-------
  - race_probabilities_fueltire.csv
  - simulation_results_fueltire.png
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
LAP_NOISE_SD = 0.30
DNF_PROB = 0.045

# --- NEW: fuel/tyre decomposition ---
FUEL_LOAD_KG = 110.0            # typical F1 race fuel load (public estimate, not
                                 # measured -- fuel telemetry isn't in the OpenF1
                                 # data pulled for this project)
FUEL_EFFECT_PER_KG = 0.03       # sec/lap per kg carried -- standard, widely-cited
                                 # F1 rule of thumb (roughly 0.3s/lap per 10kg)
FUEL_BURN_PER_LAP = FUEL_LOAD_KG / RACE_LAPS
FUEL_EFFECT_RATE = FUEL_BURN_PER_LAP * FUEL_EFFECT_PER_KG   # sec/lap FASTER each
                                                              # successive lap

NET_MEDIUM_SLOPE = 0.011        # real fitted net slope (see original script)
NET_HARD_SLOPE = -0.004
MEDIUM_DEG = NET_MEDIUM_SLOPE + FUEL_EFFECT_RATE   # corrected, tyre-only rate
HARD_DEG = NET_HARD_SLOPE + FUEL_EFFECT_RATE       # corrected, tyre-only rate

print(f"Fuel burn: {FUEL_BURN_PER_LAP:.3f} kg/lap -> fuel effect: {FUEL_EFFECT_RATE:.4f} s/lap faster")
print(f"Corrected tyre degradation -- Medium: {MEDIUM_DEG:.4f} s/lap (was {0.012} net in the original model)")
print(f"Corrected tyre degradation -- Hard:   {HARD_DEG:.4f} s/lap (was {0.010} net in the original model)\n")

N_RUNS = 20000

# ---------------------------------------------------------------------------
# 3. MONTE CARLO SIMULATION -- identical strategy/SC/DNF logic to the original,
#    only the degradation math changes
# ---------------------------------------------------------------------------
finish_pos = np.zeros((N_RUNS, n), dtype=int)
stop_count = np.zeros((N_RUNS, n), dtype=int)

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

    # --- Tyre degradation, using the CORRECTED (fuel-separated) rates ---
    stint1 = pit1_lap
    two_stop = pit2_lap >= 0
    stint2 = np.where(two_stop, pit2_lap - pit1_lap, RACE_LAPS - pit1_lap)
    stint3 = np.where(two_stop, RACE_LAPS - pit2_lap, 0)
    tyre_deg = (MEDIUM_DEG * stint1 * (stint1 + 1) / 2
                + HARD_DEG * stint2 * (stint2 + 1) / 2
                + HARD_DEG * stint3 * (stint3 + 1) / 2)

    # --- Fuel effect: identical for every driver (same fuel load/burn assumed),
    #     included for transparency -- this term is a constant offset and does
    #     NOT change relative finishing order or any probability ---
    fuel_effect_total = -FUEL_EFFECT_RATE * RACE_LAPS * (RACE_LAPS + 1) / 2

    noise = rng.normal(0, LAP_NOISE_SD * np.sqrt(RACE_LAPS), size=n)
    total_time = base_pace_gap * RACE_LAPS + tyre_deg + fuel_effect_total + pit_loss + noise

    dnf = rng.random(n) < DNF_PROB
    retirement_lap = rng.integers(1, RACE_LAPS, size=n)
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

results = pd.DataFrame({
    "driver": names, "team": teams, "grid": grid,
    "win_prob_%": (win_prob * 100).round(2),
    "podium_prob_%": (podium_prob * 100).round(2),
    "points_prob_%": (points_prob * 100).round(2),
    "avg_sim_finish": avg_finish.round(2),
    "avg_stops": avg_stops.round(2),
    "actual_finish": [actual_finish[nm] for nm in names],
}).sort_values("win_prob_%", ascending=False).reset_index(drop=True)

results.to_csv("race_probabilities_fueltire.csv", index=False)
print(results.to_string(index=False))

actual_numeric = results["actual_finish"].apply(lambda v: v if isinstance(v, (int, float)) else n + 1)
spearman = results["avg_sim_finish"].corr(actual_numeric, method="spearman")
print(f"\nSpearman correlation (pre-race expected finish vs actual finish): {spearman:.3f}")

# ---------------------------------------------------------------------------
# 5. STRATEGY COMPARISON with the corrected degradation rates
# ---------------------------------------------------------------------------
def stint_time(stints, deg_rates):
    return sum(d * L * (L + 1) / 2 for L, d in zip(stints, deg_rates))

one_stop_stints = [26, 26]
one_stop_total = stint_time(one_stop_stints, [MEDIUM_DEG, HARD_DEG]) + PIT_LOSS_GREEN

two_stop_stints = [17, 17, 18]
two_stop_total = stint_time(two_stop_stints, [MEDIUM_DEG, HARD_DEG, HARD_DEG]) + 2 * PIT_LOSS_GREEN

delta = two_stop_total - one_stop_total
print(f"\nGreen-flag strategy comparison, CORRECTED tyre degradation (no SC):")
print(f"  1-stop total cost: {one_stop_total:.1f}s")
print(f"  2-stop total cost: {two_stop_total:.1f}s")
print(f"  1-stop is {delta:.1f}s faster (vs. the original model's 18.4s gap)")
print(f"  Note the direction: the ABSOLUTE cost of both strategies rose a lot (roughly "
      f"29s/47s in the original model vs {one_stop_total:.0f}s/{two_stop_total:.0f}s here), "
      f"but the GAP between them actually narrowed. That's because degradation cost scales "
      f"with the square of stint length -- a 2-stop's three shorter stints benefit more from "
      f"higher wear rates than a 1-stop's two longer ones, partly offsetting the extra pit "
      f"stop. Net effect: tyre management matters far more in absolute terms now, but the "
      f"1-stop-vs-2-stop call itself is closer than the original model suggested.")

# ---------------------------------------------------------------------------
# 6. CHARTS
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
ax.set_title("Fuel/tyre-separated model\n2026 British GP (20,000 simulations)")
ax.legend()

ax = axes[1]
labels = ["1-stop\n(C2→C1)", "2-stop\n(C2→C1→C1)"]
values = [one_stop_total, two_stop_total]
bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
ax.set_ylabel("Tyre deg + pit-stop cost (sec, green flag)")
ax.set_title(f"Strategy comparison, corrected tyre deg\n1-stop faster by {delta:.1f}s")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}s", ha="center")

plt.tight_layout()
plt.savefig("simulation_results_fueltire.png", dpi=150)
print("\nSaved race_probabilities_fueltire.csv and simulation_results_fueltire.png")
