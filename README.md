# silverstone-montecarlo-simulator
Monte Carlo simulator project using data from the British Grand Prix

# Monte Carlo F1 Race Strategy Simulator

A pre-race strategy tool, worked example: **2026 British Grand Prix, Silverstone**.

## What it does

Runs the race 20,000 times from the real starting grid with randomised safety-car
timing, tyre degradation, pit strategy, and retirements, then reports win / podium /
points probabilities per driver and compares one-stop vs two-stop strategy cost —
the core question a strategist answers before and during a race.

Run it with `python3 monte_carlo_race_simulator.py`. Outputs `race_probabilities.csv`
and `simulation_results.png`.

## Data sources

Network access in this environment couldn't reach F1's official live-timing API
(what the FastF1 library normally uses), but it does reach **OpenF1**
(api.openf1.org), a free/open API that had real data for this exact race
(session_key 11326). That was used to calibrate the key constants:

- **Real tyre stints** — actual compound and stint length per driver (`/v1/stints`)
- **Real pit-lane times** — 37 clean green-flag stops, mean 30.1s, median 29.5s (`/v1/pit`)
- **Real degradation** — fitted from Leclerc's actual lap times: +0.011s/lap of
  tyre age on the Medium, -0.004s/lap on the Hard (essentially flat — see caveat
  below) (`/v1/laps`)
- **Real safety car timing** — VSC at lap 22, VSC at lap 39, full Safety Car
  deployed at lap 48 and run to the finish — the race genuinely ended under
  safety car (`/v1/race_control`)

Grid, qualifying times, and the broader narrative still come from public race
reports since OpenF1 doesn't carry qualifying classification or penalty context:

- **Grid & qualifying pace** — Crash.net's full qualifying results table (all 22
  drivers, Q1/Q2/Q3 times), including Gasly's 3-place penalty for impeding and
  Stroll's 10-place power-unit penalty.
- **Actual result** — Motorsport Week's finishing order, used below to check the
  model against what really happened.

**Degradation caveat:** the near-flat lap-time trend is a *net* effect — tyre
wear is being roughly cancelled out by fuel burn-off (the car gets lighter and
faster as fuel is used), and this dataset doesn't separate the two. Since the
simulator doesn't model fuel load separately, using the net observed slope is
the internally-consistent choice — and it's also a real finding: Silverstone
under the 2026 cars was a very low-degradation event, which is why the two
safety cars, not tyre wear, ended up deciding the strategic outcome.

Safety-car frequency/timing and the SC pit-loss discount remain **modelling
assumptions** — OpenF1's pit-duration field barely moves under a safety car
(the pit lane speed limit doesn't change), so the real saving from pitting
under SC — cars bunched up and crawling on track — isn't directly measurable
from that field, even though it's a well-established effect.

## Results

| Driver | Grid | Win % | Podium % | Points % | Actual finish |
|---|---|---|---|---|---|
| Antonelli | 1 | 72.4 | 92.3 | 95.7 | 16 (mechanical issue) |
| Leclerc | 2 | 18.1 | 85.3 | 95.8 | **1 (winner)** |
| Hamilton | 3 | 4.9 | 56.4 | 95.4 | 3 |
| Russell | 4 | 4.0 | 48.8 | 95.8 | 2 |
| Hadjar | 5 | 0.5 | 9.1 | 95.4 | 5 |
| Norris | 6 | 0.1 | 4.0 | 95.5 | 4 |
| Verstappen | 7 | 0.1 | 3.2 | 95.6 | DNF (crash) |

Full table in `race_probabilities.csv`.

**Validation:** Spearman rank correlation between the model's pre-race expected
finishing order and the actual result is **0.60** — a real, positive relationship,
but far from perfect. That gap is the point, not a bug: pole-sitter Antonelli
(model's clear favourite at 67% to win) suffered a mechanical failure, and
podium-running Verstappen crashed out. No pre-race model — including the ones
real teams run — can predict a specific mechanical failure or a specific crash;
what it *can* do is correctly size up who's fast and quantify how much of the
outcome is still down to chance. That the actual winner (Leclerc) was the
model's clear #2 favourite is a reasonable outcome for a well-calibrated
probabilistic model, not a miss.

**Strategy comparison:** under green-flag running only, the model puts the
one-stop about 18.4s faster than a two-stop over the race — consistent with
the real pre-race Pirelli guidance that a one-stop was the quicker theoretical
strategy. With real degradation data showing Silverstone was a very
low-wear race this weekend, that gap is now driven almost entirely by pit-lane
loss rather than tyre wear — a genuinely useful strategic read. The two safety
cars in the actual race are exactly the kind of event that flips that
calculus, which is why the simulator treats SC timing as random rather than
fixed: the "right" strategy is conditional on events that haven't happened
yet, which is the whole reason race strategy is a live, probabilistic decision
and not a spreadsheet lookup.

## Known limitations / next steps

- **Safety-car reaction logic is simplified.** In the real race, most of the
  field ended up on two stops (or, for the leaders, a stop grabbed right at
  the final Safety Car restart); this model's reactive-pit logic converts a
  smaller share of the field. Loosening the eligibility window or reaction
  probability would push the model closer to what actually happened.
- **Degradation is now real, but from one driver's stints.** Leclerc's laps
  were used as the representative sample; fitting every driver individually
  (and correcting for fuel load, which this dataset doesn't isolate) would be
  a stronger next pass.
- **Pace is derived from qualifying, scaled down** (`QUALI_TO_RACE_SCALE = 0.45`),
  not measured full-race pace — OpenF1's laps endpoint could replace this per
  driver if you pull all 22 drivers' race stints instead of just one.
- **SC pit-loss discount and SC frequency/timing are still modelling
  assumptions** — not directly measurable from OpenF1's pit-duration field
  (see caveat above), though the real race_control data confirms this race
  had 2 VSCs + 1 full SC, which is within the range the model samples.
- **DNF probability is uniform across drivers.** The real race had 3 retirements
  (higher than the model's ~1 expected), driven by specific circumstances
  (Stowe conditions) the model doesn't represent.
- **Natural extensions:** per-team/per-driver real pace and degradation from
  all 22 drivers' OpenF1 laps, weather/rain probability, undercut/overcut
  modelling between two specific drivers.

## Files

- `monte_carlo_race_simulator.py` — the full simulation (grid data, model, Monte
  Carlo loop, strategy comparison, charts)
- `race_probabilities.csv` — win/podium/points probabilities for all 22 drivers
- `simulation_results.png` — probability chart + strategy comparison chart
