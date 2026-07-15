"""
Monte Carlo F1 Race Strategy Simulator
=======================================
Case study: 2026 British Grand Prix (Silverstone), Round 9.

WHAT THIS DOES
---------------
Simulates the race thousands of times from the real starting grid, with
randomised safety-car timing, tyre degradation, pit-stop strategy, and
mechanical/incident retirements, to produce win / podium / points
probabilities for every driver and to compare one-stop vs two-stop
strategy under uncertainty -- the kind of pre-race and live strategy
question a race strategist actually has to answer.

DATA SOURCES / LIMITATIONS
---------------------------
This environment's network access does not reach F1's official live-timing
API (the backend FastF1 normally uses), but it DOES reach the free/open
OpenF1 API (api.openf1.org), which was used to pull real data for this
race (session_key=11326, meeting_key=1289) and calibrate the constants
below:
  - Real tyre stints per driver (compound, stint length, tyre age) --
    `/v1/stints`
  - Real pit-lane transit times -- `/v1/pit` (clean green-flag stops:
    n=37, mean 30.1s, median 29.5s, std 1.9s)
  - Real lap-by-lap timing for a representative driver (Leclerc, #16) --
    `/v1/laps`, used to fit an actual degradation slope per compound
  - Real safety-car / VSC timing -- `/v1/race_control`: VSC at lap 22
    (1 lap), VSC at lap 39 (1 lap), full Safety Car deployed at lap 48
    and run to the finish (laps 48-52) -- the race genuinely ended
    under safety car, matching the "finishes under safety car" headlines.

Grid, qualifying times, and the broader race narrative still come from
public race reports (Crash.net, Motorsport Week, Formula1.com, 04-05 Jul
2026), since OpenF1 doesn't carry qualifying classification or penalty
context in one place.

IMPORTANT CAVEAT ON DEGRADATION: fitting Leclerc's clean green-flag laps
gave a near-flat trend (Medium: +0.011s/lap of tyre age; Hard: -0.004s/lap,
statistically indistinguishable from flat). This is a net effect --
tyre wear is being roughly cancelled out by fuel burn-off (the car gets
lighter, and therefore faster, as the race goes on), and this dataset
doesn't isolate the two. Since this simulator doesn't model fuel load
separately, using the net observed slope is the internally-consistent
choice, and it also tells you something real about this race: Silverstone
under the 2026 cars was a very low-degradation event, which is why the
2 safety cars -- not tyre wear -- were what actually decided strategy.
PIT_LOSS_GREEN is derived from the measured 30.1s raw pit-lane time minus
a track-pace-equivalent offset (the time a driver would have spent
covering that distance at racing speed instead), consistent with
Silverstone's commonly published ~20-21s pit loss figure.
PIT_LOSS_SC and the safety-car frequency/timing distribution remain
modelling assumptions -- OpenF1's pit-duration field doesn't change much
under a safety car (pit lane speed limit is the same), so the real
saving from pitting under SC (the field bunching up on track) isn't
directly measurable from this field; it's a well-established but not
directly measured effect here.

OUTPUTS
-------
  - race_probabilities.csv   win / podium / points probabilities per driver
  - simulation_results.png   probability charts + strategy comparison
  - Printed strategy comparison (1-stop vs 2-stop) for the race leader
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. REAL STARTING GRID -- 2026 British Grand Prix, Silverstone
#    (name, team, grid position post-penalties, qualifying best lap in sec)
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
    ("Gasly",       "Alpine",       15, 90.063),   # penalised from P12 (impeding)
    ("Albon",       "Williams",     16, 90.743),
    ("Ocon",        "Haas",         17, 90.680),
    ("Bottas",      "Cadillac",     18, 91.227),
    ("Colapinto",   "Alpine",       19, 91.321),
    ("Perez",       "Cadillac",     20, 91.940),
    ("Alonso",      "Aston Martin", 21, 93.025),
    ("Stroll",      "Aston Martin", 22, 92.863),   # penalised from P21 (PU elements)
]

# Actual 2026 British GP finishing order, for validating the model against
# what really happened (Motorsport Week race report). Non-finishers are shown
# as their classification status (DNF/DNS/DSQ) rather than a numeric position.
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
# 2. MODEL ASSUMPTIONS (documented -- tune these against real data if you
#    have telemetry access)
# ---------------------------------------------------------------------------
QUALI_TO_RACE_SCALE = 0.45   # race-pace gaps are usually smaller than 1-lap quali gaps
base_pace_gap = quali_gap * QUALI_TO_RACE_SCALE   # sec/lap slower than the pace leader

RACE_LAPS = 52                # British GP race distance
PIT_LOSS_GREEN = 21.0         # sec, net strategic loss -- derived from real OpenF1 pit
                               # data (37 clean stops, mean raw pit-lane time 30.1s) minus
                               # a track-pace-equivalent offset; matches Silverstone's
                               # commonly published ~20-21s pit loss
PIT_LOSS_SC = 7.0             # sec, reduced loss pitting under Safety Car / VSC
                               # (not directly measurable from OpenF1's pit-duration field;
                               # standard industry estimate, see caveat above)
MEDIUM_DEG = 0.012            # sec lost per lap of age, Medium -- real fitted slope from
                               # Leclerc's green-flag Medium stint (laps 2-25) was +0.011s/lap
HARD_DEG = 0.010              # sec lost per lap of age, Hard -- real fitted slope from
                               # Leclerc's Hard stint (laps 27-47) was -0.004s/lap (flat,
                               # within noise); floored to a small positive value since
                               # tyres can't get faster with age in this model
LAP_NOISE_SD = 0.30           # sec, per-lap race-pace variance (traffic, small errors)
DNF_PROB = 0.045              # per-driver probability of a race-ending issue

N_RUNS = 20000

# ---------------------------------------------------------------------------
# 3. MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------
finish_pos = np.zeros((N_RUNS, n), dtype=int)
stop_count = np.zeros((N_RUNS, n), dtype=int)

for r in range(N_RUNS):
    # --- Random safety-car periods this race ---
    n_sc = rng.choice([0, 1, 2, 3], p=[0.10, 0.25, 0.40, 0.25])
    sc_laps = np.sort(rng.choice(np.arange(10, 48), size=n_sc, replace=False)) if n_sc else np.array([])

    # --- Planned green-flag one-stop pit lap per driver ---
    planned_pit = rng.integers(23, 33, size=n)

    # --- First stop: taken opportunistically under SC1 if it lands close to
    #     the driver's planned window, otherwise taken green at the planned lap ---
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

    # --- Optional cheap second stop under SC2, only for drivers who already
    #     banked an early SC1 stop and have enough tyre life to make it worth it ---
    if n_sc >= 2:
        sc2 = sc_laps[1]
        eligible2 = pitted_under_sc & ((sc2 - pit1_lap) >= 8)
        react2 = eligible2 & (rng.random(n) < 0.75)
        pit_loss = np.where(react2, pit_loss + PIT_LOSS_SC, pit_loss)
        stops = np.where(react2, 2, stops)
        pit2_lap = np.where(react2, sc2, pit2_lap)

    # --- Tyre degradation: triangular time-loss over each stint ---
    stint1 = pit1_lap
    two_stop = pit2_lap >= 0
    stint2 = np.where(two_stop, pit2_lap - pit1_lap, RACE_LAPS - pit1_lap)
    stint3 = np.where(two_stop, RACE_LAPS - pit2_lap, 0)

    deg = (MEDIUM_DEG * stint1 * (stint1 + 1) / 2
           + HARD_DEG * stint2 * (stint2 + 1) / 2
           + HARD_DEG * stint3 * (stint3 + 1) / 2)

    # --- Total race time relative to a theoretical zero-deg, zero-stop pace leader ---
    noise = rng.normal(0, LAP_NOISE_SD * np.sqrt(RACE_LAPS), size=n)
    total_time = base_pace_gap * RACE_LAPS + deg + pit_loss + noise

    # --- Retirements ---
    dnf = rng.random(n) < DNF_PROB
    retirement_lap = rng.integers(1, RACE_LAPS, size=n)
    sort_key = np.where(dnf, 1_000_000 - retirement_lap, total_time)

    order = np.argsort(sort_key)
    finish_pos[r, order] = np.arange(1, n + 1)
    stop_count[r, :] = stops

# ---------------------------------------------------------------------------
# 4. AGGREGATE RESULTS
# ---------------------------------------------------------------------------
win_prob = (finish_pos == 1).mean(axis=0)
podium_prob = (finish_pos <= 3).mean(axis=0)
points_prob = (finish_pos <= 10).mean(axis=0)
avg_finish = finish_pos.mean(axis=0)
avg_stops = stop_count.mean(axis=0)

results = pd.DataFrame({
    "driver": names,
    "team": teams,
    "grid": grid,
    "win_prob_%": (win_prob * 100).round(2),
    "podium_prob_%": (podium_prob * 100).round(2),
    "points_prob_%": (points_prob * 100).round(2),
    "avg_sim_finish": avg_finish.round(2),
    "avg_stops": avg_stops.round(2),
    "actual_finish": [actual_finish[nm] for nm in names],
}).sort_values("win_prob_%", ascending=False).reset_index(drop=True)

results.to_csv("race_probabilities.csv", index=False)
print(results.to_string(index=False))

# For the rank correlation below, DNF/DNS/DSQ aren't numeric positions, so
# treat any non-finisher as classified one place behind the last finisher.
actual_numeric = results["actual_finish"].apply(lambda v: v if isinstance(v, (int, float)) else n + 1)
spearman = results["avg_sim_finish"].corr(actual_numeric, method="spearman")
print(f"\nSpearman correlation (pre-race expected finish vs actual finish): {spearman:.2f}")

# ---------------------------------------------------------------------------
# 5. STRATEGY COMPARISON: 1-stop vs 2-stop, green-flag only (no safety cars)
#    -- this reproduces the pre-race "which strategy is faster" question
#    strategists answer before a race even starts.
# ---------------------------------------------------------------------------
def stint_time(stints, deg_rates):
    return sum(d * L * (L + 1) / 2 for L, d in zip(stints, deg_rates))

one_stop_stints = [26, 26]
one_stop_deg = stint_time(one_stop_stints, [MEDIUM_DEG, HARD_DEG])
one_stop_total = one_stop_deg + PIT_LOSS_GREEN

two_stop_stints = [17, 17, 18]
two_stop_deg = stint_time(two_stop_stints, [MEDIUM_DEG, HARD_DEG, HARD_DEG])
two_stop_total = two_stop_deg + 2 * PIT_LOSS_GREEN

delta = two_stop_total - one_stop_total
print(f"\nGreen-flag strategy comparison (tyre deg + pit loss only, no SC):")
print(f"  1-stop total cost: {one_stop_total:.1f}s")
print(f"  2-stop total cost: {two_stop_total:.1f}s")
print(f"  1-stop is {delta:.1f}s faster under green-flag running "
      f"(matches the pre-race strategy call that a 1-stop was quickest; "
      f"the two safety-car periods in the real race are what flipped this)")

# ---------------------------------------------------------------------------
# 6. BOOTSTRAP CONFIDENCE INTERVALS -- the 20,000-run estimate is itself a
#    sample; resample it to show how much sampling noise is left in the
#    win/podium/points percentages (distinct from the race's own randomness).
# ---------------------------------------------------------------------------
N_BOOT = 500
boot_win = np.zeros((N_BOOT, n))
boot_podium = np.zeros((N_BOOT, n))
boot_points = np.zeros((N_BOOT, n))
for b in range(N_BOOT):
    idx = rng.integers(0, N_RUNS, size=N_RUNS)
    sample = finish_pos[idx]
    boot_win[b] = (sample == 1).mean(axis=0) * 100
    boot_podium[b] = (sample <= 3).mean(axis=0) * 100
    boot_points[b] = (sample <= 10).mean(axis=0) * 100

ci_lo_win = np.percentile(boot_win, 2.5, axis=0)
ci_hi_win = np.percentile(boot_win, 97.5, axis=0)
ci_lo_pod = np.percentile(boot_podium, 2.5, axis=0)
ci_hi_pod = np.percentile(boot_podium, 97.5, axis=0)
ci_lo_pts = np.percentile(boot_points, 2.5, axis=0)
ci_hi_pts = np.percentile(boot_points, 97.5, axis=0)

order_idx = [names.index(d) for d in results["driver"]]

# ---------------------------------------------------------------------------
# 7. CHARTS
# ---------------------------------------------------------------------------
top10 = results.head(10)
top10_idx = [names.index(d) for d in top10["driver"]]

fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])

# --- Win/podium/points with bootstrap 95% CI error bars ---
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(top10))
width = 0.25
werr = np.abs(np.vstack([top10["win_prob_%"] - ci_lo_win[top10_idx], ci_hi_win[top10_idx] - top10["win_prob_%"]]))
perr = np.abs(np.vstack([top10["podium_prob_%"] - ci_lo_pod[top10_idx], ci_hi_pod[top10_idx] - top10["podium_prob_%"]]))
qerr = np.abs(np.vstack([top10["points_prob_%"] - ci_lo_pts[top10_idx], ci_hi_pts[top10_idx] - top10["points_prob_%"]]))
ax.bar(x - width, top10["win_prob_%"], width, yerr=werr, capsize=3, label="Win %")
ax.bar(x, top10["podium_prob_%"], width, yerr=perr, capsize=3, label="Podium %")
ax.bar(x + width, top10["points_prob_%"], width, yerr=qerr, capsize=3, label="Points %")
ax.set_xticks(x)
ax.set_xticklabels(top10["driver"], rotation=45, ha="right")
ax.set_ylabel("Probability (%)")
ax.set_title("Pre-race probabilities, with 95% bootstrap CI\n2026 British GP (20,000 simulations)")
ax.legend(fontsize=8)

# --- Strategy comparison ---
ax = fig.add_subplot(gs[0, 1])
labels = ["1-stop\n(C2→C1)", "2-stop\n(C2→C1→C1)"]
values = [one_stop_total, two_stop_total]
bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
ax.set_ylabel("Tyre deg + pit-stop cost (sec, green flag)")
ax.set_title("Strategy comparison (no safety car)\nLower = faster")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}s", ha="center")

# --- Finishing-position probability heatmap (all drivers x all positions) ---
ax = fig.add_subplot(gs[1, :])
pos_prob = np.zeros((n, n))
for i in range(n):
    for p in range(1, n + 1):
        pos_prob[i, p - 1] = (finish_pos[:, i] == p).mean() * 100

heat = pos_prob[order_idx]
im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(20, heat.max()))
ax.set_xticks(np.arange(n))
ax.set_xticklabels(np.arange(1, n + 1))
ax.set_yticks(np.arange(n))
ax.set_yticklabels(results["driver"])
ax.set_xlabel("Finishing position")
ax.set_title("Full finishing-position probability distribution (%), all drivers, all positions")
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label("Probability (%)")

plt.tight_layout()
plt.savefig("simulation_results.png", dpi=150)
print("\nSaved race_probabilities.csv and simulation_results.png")
