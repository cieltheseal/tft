from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split

### Transformation

class DropEmptyColumns(BaseEstimator, TransformerMixin):
    # Drops columns that are empty and have no apparent purpose
    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns if not col.endswith('_name')]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

class DropCompanion(BaseEstimator, TransformerMixin):
    # Drops LL data
    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns if 'companion' not in col]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

class DropSkill(BaseEstimator, TransformerMixin):
    # Drops Remix skill tree data
    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns if 'skill' not in col]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

class DropNonFirst(BaseEstimator, TransformerMixin):
    # Drops all boards that did not place first
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[X['placement'] == 1]

class CountTacticianItems(BaseEstimator, TransformerMixin):
    # Count number of Tacticians' items on each board
    def fit(self, X, y=None):
        self.item_columns_ = [col for col in X.columns if 'itemNames' in col]
        return self

    def transform(self, X):
        X = X.copy()
        X['tactician_count'] = X[self.item_columns_].apply(
            lambda row: sum('Tactician' in str(val) for val in row),
            axis=1
        )
        return X

class CalculateBoardSize(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['board_size'] = X['level'] + X['tactician_count']
        return X

class DropCompanion(BaseEstimator, TransformerMixin):
    # Drops LL data
    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns if 'companion' not in col]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

class DropId(BaseEstimator, TransformerMixin):
    # Drops info about player account
    def __init__(self):
        self.columns_to_drop = ['puuid', 'riotIdGameName', 'riotIdTagline']

    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns if col not in self.columns_to_drop]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

class DropIrrelevant(BaseEstimator, TransformerMixin):
    # Drops irrelevant data
    def __init__(self):
        self.columns_to_drop = ['gold_left', 'last_round', 'missions_PlayerScore2', 'placement', 'players_eliminated',
                                'time_eliminated', 'total_damage_to_players', 'win', 'tactician_count']

    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns if col not in self.columns_to_drop]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

class DropTraits(BaseEstimator, TransformerMixin):
    # Drops trait data
    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns if 'trait' not in col]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

class DropItems(BaseEstimator, TransformerMixin):
    # Drops item data
    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns if 'item' not in col]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

class DropTierRarity(BaseEstimator, TransformerMixin):
    # Drops tier(*-level) and rarity(cost)
    def fit(self, X, y=None):
        self.columns_to_keep_ = [col for col in X.columns
                                 if 'tier' not in col.lower() and 'rarity' not in col.lower()]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_].copy()

class RemoveTFTPrefix(BaseEstimator, TransformerMixin):
    # Removes any "TFT[number]_" prefix from strings and lists of strings
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        pattern = re.compile(r"TFT\d+_")

        def remove_prefix(value):
            if isinstance(value, list):
                return [pattern.sub("", v) if isinstance(v, str) else v for v in value]
            elif isinstance(value, str):
                return pattern.sub("", value)
            return value

        return X.map(remove_prefix)

def generate_unit_training_pairs(df, val_size=0.2, random_state=42):
    """
    Generate training and validation pairs from unit boards.

    - Training: all possible (input_units, label_unit) pairs per board.
    - Validation: only one randomly selected (input_units, label_unit) pair per board.

    Parameters:
        df (pd.DataFrame): Input DataFrame with unit columns.
        val_size (float): Proportion of boards used for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        (train_df, val_df): Tuple of DataFrames with ['input_units', 'label_unit']
    """
    random.seed(random_state)

    unit_columns = [col for col in df.columns if col.startswith("units_") and col.endswith("_id")]

    # Split original boards into train and validation boards
    train_boards, val_boards = train_test_split(df, test_size=val_size, random_state=random_state)

    training_data = []
    validation_data = []

    # Training: generate all possible label/input pairs
    for _, row in train_boards.iterrows():
        units = [row[col] for col in unit_columns if pd.notna(row[col])]
        for i, label_unit in enumerate(units):
            if "Summon" in str(label_unit):
                continue
            input_units = units[:i] + units[i+1:]
            training_data.append({
                "input_units": input_units,
                "label_unit": label_unit
            })

    # Validation: only one random label/input pair per board
    for _, row in val_boards.iterrows():
        units = [row[col] for col in unit_columns if pd.notna(row[col])]
        if len(units) < 2:
            continue  # Skip boards too small to split
        idx = random.randint(0, len(units) - 1)
        label_unit = units[idx]
        if "Summon" in str(label_unit):
            continue
        input_units = units[:idx] + units[idx+1:]
        validation_data.append({
            "input_units": input_units,
            "label_unit": label_unit
        })

    train_df = pd.DataFrame(training_data).reset_index(drop=True)
    val_df = pd.DataFrame(validation_data).reset_index(drop=True)

    return train_df, val_df
