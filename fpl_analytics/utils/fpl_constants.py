"""FPL scoring rules, position codes, and other constants."""

# FPL position IDs
POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

POSITION_ID = {v: k for k, v in POSITION_MAP.items()}

# FPL points system
POINTS = {
    # Appearance
    "appearance_lt60": 1,
    "appearance_ge60": 2,
    # Goals
    "goal_gk": 10,
    "goal_def": 6,
    "goal_mid": 5,
    "goal_fwd": 4,
    # Assists
    "assist": 3,
    # Clean sheets
    "clean_sheet_gk": 4,
    "clean_sheet_def": 4,
    "clean_sheet_mid": 1,
    "clean_sheet_fwd": 0,
    # Goals conceded (per 2, GK/DEF only)
    "goals_conceded_2": -1,
    # Saves (per 3, GK only)
    "saves_3": 1,
    # Penalties
    "penalty_miss": -2,
    "penalty_save": 5,
    # Cards
    "yellow_card": -1,
    "red_card": -3,
    # Other
    "own_goal": -2,
    # Bonus
    "bonus_1": 1,
    "bonus_2": 2,
    "bonus_3": 3,
}

# Goal points by position shorthand
GOAL_POINTS = {
    "GK": POINTS["goal_gk"],
    "DEF": POINTS["goal_def"],
    "MID": POINTS["goal_mid"],
    "FWD": POINTS["goal_fwd"],
}

# Clean sheet points by position shorthand
CS_POINTS = {
    "GK": POINTS["clean_sheet_gk"],
    "DEF": POINTS["clean_sheet_def"],
    "MID": POINTS["clean_sheet_mid"],
    "FWD": POINTS["clean_sheet_fwd"],
}

# Valid formations (defenders, midfielders, forwards)
VALID_FORMATIONS = [
    (3, 5, 2),
    (3, 4, 3),
    (4, 5, 1),
    (4, 4, 2),
    (4, 3, 3),
    (5, 4, 1),
    (5, 3, 2),
    (5, 2, 3),
]

# Squad constraints
SQUAD_SIZE = 15
STARTING_XI = 11
SQUAD_COMPOSITION = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

# football-data.co.uk season codes (most recent 5)
SEASON_CODES = [
    "2425",
    "2324",
    "2223",
    "2122",
    "2021",
]

# Team name normalisation: maps common variants to a canonical name
TEAM_NAME_MAP = {
    # FPL API names → canonical
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle",
    "Nott'm Forest": "Nottingham Forest",
    "Southampton": "Southampton",
    "Spurs": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton",
    # FPL API short codes (fallback in case short_name is used anywhere)
    "ARS": "Arsenal",
    "AVL": "Aston Villa",
    "BOU": "Bournemouth",
    "BRE": "Brentford",
    "BHA": "Brighton",
    "BUR": "Burnley",
    "CHE": "Chelsea",
    "CRY": "Crystal Palace",
    "EVE": "Everton",
    "FUL": "Fulham",
    "IPS": "Ipswich",
    "LEI": "Leicester",
    "LIV": "Liverpool",
    "MCI": "Manchester City",
    "MUN": "Manchester United",
    "NEW": "Newcastle",
    "NFO": "Nottingham Forest",
    "SOU": "Southampton",
    "TOT": "Tottenham",
    "WHU": "West Ham",
    "WOL": "Wolverhampton",
    # football-data.co.uk variants
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham Forest",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton": "Wolverhampton",
    "Wolverhampton Wanderers": "Wolverhampton",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Burnley": "Burnley",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Watford": "Watford",
    "Norwich": "Norwich",
    "Norwich City": "Norwich",
    "West Brom": "West Brom",
    "West Bromwich Albion": "West Brom",
}


def normalise_team_name(name: str) -> str:
    """Normalise a team name to its canonical form."""
    return TEAM_NAME_MAP.get(name, name)
