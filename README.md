# FPL Analytics Engine

A Fantasy Premier League projection and optimisation platform built on the Dixon-Coles (1997) Poisson regression model.

## Features

- **Player projections** — expected points per player per gameweek, driven by a fitted Dixon-Coles model and per-90 xG/xA rates from the FPL API and Understat
- **Fixture difficulty** — attack/defence-adjusted difficulty ratings for every team across upcoming gameweeks
- **Team strengths** — attack and defence ratings for all 20 Premier League clubs, updated from live match results
- **Squad optimiser** — integer linear programme (PuLP / CBC) that selects the optimal 15-man squad within FPL budget and formation constraints
- **Transfer planner** — recommends the best 1–3 transfers from your live FPL squad, accounting for the 4-point hit cost
- **Price change alerts** — XGBoost classifier that predicts price rises and falls from net transfer rates, ownership, form, and price momentum; falls back to heuristic threshold alerts

## Stack

| Layer | Technology |
|---|---|
| Match prediction model | Dixon-Coles bivariate Poisson (scipy SLSQP / PyTorch LBFGS on GPU) |
| Player projections | Vectorised numpy, per-90 Understat xG/xA rates |
| Squad/transfer optimiser | PuLP integer LP, CBC solver |
| Price change model | XGBoost multiclass classifier (`device='cuda'`) |
| Data sources | FPL Official API, football-data.co.uk, Understat |
| Storage | SQLAlchemy + SQLite |
| Dashboard | Streamlit multi-page app |

## Installation

Requires Python 3.11+.

```bash
# Clone the repo
git clone https://github.com/george-southall/fpl-analytics.git
cd fpl-analytics

# Install dependencies
pip install -e .

# Dev dependencies (pytest, ruff)
pip install -e ".[dev]"

# CBC solver (required for squad optimiser)
# macOS
brew install coin-or-tools
# Ubuntu / Debian
sudo apt-get install coinor-cbc
# Windows — install via conda or download from https://github.com/coin-or/Cbc
```

### GPU acceleration (optional)

Dixon-Coles fitting and the XGBoost price model can use CUDA. Requires PyTorch 2.7+ (Blackwell / CUDA 12.8):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[gpu]"
```

When a CUDA device is detected, Dixon-Coles fitting automatically uses PyTorch LBFGS instead of scipy SLSQP, and XGBoost trains with `device='cuda'`. Both fall back silently to CPU if PyTorch is not installed.

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```
FPL_TEAM_ID=123456          # Your FPL team ID (found in your FPL team page URL)
FPL_DC_TIME_DECAY_XI=0.0065 # Dixon-Coles time decay (default 0.0065)
FPL_SQUAD_BUDGET=100.0      # Squad budget in £m
FPL_PROJECTION_HORIZON_GWS=6
```

All settings can also be passed as environment variables with the `FPL_` prefix.

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501`. The first load fits the Dixon-Coles model on historical results (20–60 seconds); subsequent page navigations use the cached model.

### Dashboard pages

| Page | Description |
|---|---|
| Player Projections | xPts table and bar chart for any GW horizon (1–6 GWs), filterable by position and price |
| Fixture Difficulty | Colour-coded difficulty calendar for all 20 clubs |
| Team Strengths | Attack / defence ratings scatter, ranked table |
| Squad Optimiser | Builds the optimal FPL squad from scratch for a given budget |
| Transfer Planner | Connects to your live FPL team and recommends transfers with net xPts gain |
| Price Change Alerts | XGBoost rise/fall/hold predictions with probability scores; retrain button in sidebar |

## Refreshing data

```bash
# Refresh FPL API data + match results
python -m fpl_analytics.ingestion.refresh

# Skip Understat (faster, no xG/xA data)
python -m fpl_analytics.ingestion.refresh --skip-understat

# FPL API only
python -m fpl_analytics.ingestion.refresh --fpl-only
```

## Running the optimiser from the CLI

```bash
# Recommend the best single transfer (1 free transfer, 1 GW horizon)
python -m fpl_analytics.optimiser.run --transfers 1 --free-transfers 1 --gws 1

# Up to 2 transfers over 3 GWs
python -m fpl_analytics.optimiser.run --transfers 2 --free-transfers 1 --gws 3
```

## Running tests

```bash
pytest
```

100 tests covering Dixon-Coles parameter recovery, FPL points calculator, squad optimiser constraints, and the price change model.

## Project structure

```
fpl_analytics/
├── config.py                  # Pydantic-settings configuration
├── db.py                      # SQLAlchemy models and session management
├── ingestion/
│   ├── fpl_api.py             # FPL API client with TTL caching
│   ├── results_fetcher.py     # football-data.co.uk historical results
│   ├── understat_fetcher.py   # Understat xG/xA data
│   ├── data_validator.py      # Cross-source validation
│   └── refresh.py             # CLI data refresh script
├── models/
│   ├── dixon_coles.py         # Dixon-Coles model (scipy + PyTorch GPU path)
│   ├── score_matrix.py        # Scoreline probability matrix and derived stats
│   └── team_strengths.py      # Team attack/defence rating table
├── projections/
│   ├── projection_engine.py   # Full player xPts projection pipeline
│   ├── points_calculator.py   # FPL scoring formula
│   ├── minutes_model.py       # Expected minutes model (with DGW support)
│   └── fixture_difficulty.py  # Fixture difficulty ratings
├── optimiser/
│   ├── squad_optimiser.py     # ILP squad selection (PuLP)
│   ├── transfer_optimiser.py  # Transfer recommendation engine
│   ├── captain_picker.py      # Captain / vice / differential picker
│   └── run.py                 # CLI entry point
└── price_changes/
    ├── net_transfers.py       # Per-GW feature engineering from API histories
    ├── price_model.py         # XGBoost price change classifier
    └── alerts.py              # Alert generation combining model + heuristics

dashboard/
├── app.py                     # Streamlit entry point
├── data_loader.py             # Cached data loading functions
└── pages/
    ├── 01_Player_Projections.py
    ├── 02_Fixture_Difficulty.py
    ├── 03_Team_Strengths.py
    ├── 04_Squad_Optimiser.py
    ├── 05_Transfer_Planner.py
    └── 06_Price_Changes.py
```

## Model details

### Dixon-Coles

The Dixon-Coles (1997) bivariate Poisson model estimates per-team attacking strength (α), defensive strength (δ), home advantage (γ), and a low-score correction term (ρ) via maximum likelihood:

```
λ = α_home · δ_away · γ     (expected home goals)
μ = α_away · δ_home          (expected away goals)
P(X=x, Y=y) = τ(x,y) · Poisson(x|λ) · Poisson(y|μ)
```

The τ correction adjusts the joint probability for scorelines 0-0, 1-0, 0-1, and 1-1, where the independence assumption is weakest. Historical matches are weighted by exponential time decay (ξ = 0.0065, ≈ half-life of ~3 months).

### Price change model

Features: `net_transfer_rate` (net transfers / ownership), `ownership_pct`, `value` (price), `price_change_lag1`, `net_rate_lag1`, rolling 5-GW form, position encoding. Target: sign of next-GW price change (+1 / 0 / −1). Training uses walk-forward cross-validation on current-season GW histories fetched from the FPL API.

## References

- Dixon, M.J. and Coles, S.G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society, Series C.
- [FPL Official API](https://fantasy.premierleague.com/api/bootstrap-static/)
- [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
- [Understat](https://understat.com)
