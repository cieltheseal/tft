import pandas as pd

def extract_unit_rarities(csv_path) -> dict:
    """
    :param df:
    :return:
    """
    # First pass: find all unique character IDs
    df_all = pd.read_csv(csv_path)
    all_char_ids = set()

    for i in range(1, 12):
        char_col = f'units_{i}_character_id'
        all_char_ids.update(df_all[char_col].dropna().unique())

    target_count = len(all_char_ids)
    collected_units = {}

    # Second pass: read rows and collect rarity until all char_ids found
    for _, row in df_all.iterrows():
        for i in range(1, 12):
            char_col = f'units_{i}_character_id'
            rarity_col = f'units_{i}_rarity'

            char_id = row.get(char_col)
            rarity = row.get(rarity_col)

            if pd.isna(char_id) or pd.isna(rarity):
                continue

            char_id = char_id.split('_', 1)[1] if '_' in char_id else char_id
            if rarity == 6:
                rarity = 5
            if rarity < 4:
                rarity += 1

            if char_id not in collected_units:
                collected_units[char_id] = int(rarity)

                if len(collected_units) == target_count:
                    return collected_units

    return collected_units

if __name__ == "__main__":
    unit_rarities = extract_unit_rarities("data/analysis.csv")
    rarity_to_level = {
        1: 1,
        2: 2,
        3: 4,
        4: 5,
        5: 7
    }
# Convert to DataFrame
df = pd.DataFrame([
    {"character": k, "rarity": v, "min_level": rarity_to_level.get(v, 0)}
    for k, v in unit_rarities.items()
])

# Save to CSV
df.to_csv("data/rarity.csv", index=False)

## Originally
## Rarity 0: 1-cost
## Rarity 1: 2-cost
## Rarity 2: 3-cost
## Rarity 4: 4-cost
## Rarity 6: 5-cost