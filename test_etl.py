from prefect import flow, task
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline

from extract import *
from transform import *

import pandas as pd
import urllib
import pyodbc
from sqlalchemy import create_engine
import concurrent.futures
import os
import time

# Define constants
platforms = ["NA1", "KR", "EUW1", "SG2"]
regions = {"Americas": ["NA1", "BR1"], "Asia": ["KR", "JP"], "Europe": ["EUW1"], "SEA": ["SG2"]}

#------------------------
# Tasks
#------------------------

@task
def run_pipeline_for_platform(platform: str, api_key: str, n_matches: int = 10):
    try:
        print(f"🔄 Starting pipeline for {platform}")
        puuids = get_challengers(platform, api_key)
        region = next((region for region, p_list in regions.items() if platform in p_list), None)
        batch_size = 5
        n_puuids = 50
        match_ids = set()
        puuids = puuids[:n_puuids]
        all_match_ids = set()

        for i in range(0, len(puuids), batch_size):
            print(f"Getting matches for {i+1} to {i+5}-th PUUIDs...")
            chunk = puuids[i:i+batch_size]
            ids = get_matches(chunk, region, n_matches, api_key)
            all_match_ids.update(ids)
            if i + batch_size < len(puuids):
                time.sleep(11)

        df = get_matches_info(all_match_ids, region, api_key)
        print(f"✅ Finished {platform}, {len(df)} matches from {n_puuids} Challengers processed.")
        return df
    except Exception as e:
        print(f"❌ Error in pipeline for {platform}: {e}")
        return pd.DataFrame()

@task
def transform_data_pipeline(df: pd.DataFrame) -> None:
    # Analysis pipeline
    pipe_analysis = Pipeline([
        ("dropempty", DropEmptyColumns()),
        ("dropcompanion", DropCompanion()),
        ("dropid", DropId()),
    ])

    df = pipe_analysis.fit_transform(df)
    df.to_csv('data/analysis.csv', index=False)
    print(f"✅ Saved analysis.csv ({df.shape})")

    # Optimizer pipeline
    pipe_opt = Pipeline([
        ("dropnonfirst", DropNonFirst()),
        ("removetftprefix", RemoveTFTPrefix()),
    ])
    df = pipe_opt.fit_transform(df)

    train, val = generate_unit_training_pairs(df)
    train.to_csv('data/train_opt.csv', index=False)
    val.to_csv('data/val_opt.csv', index=False)

    print(f"✅ Saved train_opt.csv ({train.shape}) and val_opt.csv ({val.shape})")

#------------------------
# Flow
#------------------------

@flow(name="etl-flow")
def etl_flow():
    load_dotenv('.env')
    api_key = os.getenv('RIOT_API_KEY')

    # Run each platform concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_pipeline_for_platform.fn, platform, api_key) for platform in platforms]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Merge all non-empty DataFrames
    dfs = [df for df in results if not df.empty]
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        transform_data_pipeline(full_df)
    else:
        print("⚠️ No data to process")

#------------------------
# Run directly
#------------------------

if __name__ == "__main__":
    etl_flow.serve(
        name="etl-deployment",
        cron="0 12 * * *"
    )
