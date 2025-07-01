from sklearn.pipeline import Pipeline
from extract import *
from transform import *

import urllib
import pyodbc
from sqlalchemy import create_engine

import concurrent.futures

from dotenv import load_dotenv
import os
import time

from prefect import flow, task

#api_key = 'RGAPI-blablabla'
platforms = ["NA1", "KR", "EUW1", "SG2"]
regions = {"Americas": ["NA1", "BR1"], "Asia": ["KR", "JP"], "Europe": ["EUW1"], "SEA": ["SG2"]}

#sql_server = "tftopt.database.windows.net"
#sql_database = "tftopt"

#------------------------------------------------
# Concurrent API calls from regional endpoints
#------------------------------------------------

def pipeline_for_server(platform, api_key, n_matches=10):
    try:
        print(f"🔄 Starting pipeline for {platform}")
        puuids = get_challengers(platform, api_key)
        region = next((region for region, platforms in regions.items() if platform in platforms), None)
        batch_size = 5
        n_puuids = 50
        match_ids = set()
        puuids = puuids[:n_puuids]
        all_match_ids = set()
        # Process PUUIDs in chunks of 5
        for i in range(0, len(puuids), batch_size):
            print(f"Getting matches for {i+1} to {i+5}-th PUUIDs...")
            puuid_chunk = puuids[i:i + 5]

            # Extract match IDs for this chunk
            match_ids = get_matches(puuid_chunk, region, n_matches, api_key)
            all_match_ids.update(match_ids)

            # Wait 10 seconds between batches
            if i + 5 < len(puuids):  # Avoid waiting after last chunk
                time.sleep(10)
        db = get_matches_info(match_ids, region, api_key)

        print(f"✅ Finished pipeline for {platform}, {len(db)} matches from {n_puuids} Challengers processed.")
        return db
    except Exception as e:
        print(f"❌ Error in pipeline for {platform}: {e}")

#------------------------------------------------
# Modular pipelines for data transformation
#------------------------------------------------

def use_data_pipeline(match_data) -> 'DataFrame':
    # Pipeline for analysis
    pipe_analysis = Pipeline([
        ("dropempty", DropEmptyColumns()),
        ("dropcompanion", DropCompanion()),
        ("dropid", DropId()),
    ])

    match_data = pipe_analysis.fit_transform(match_data)
    match_data.to_csv('data/analysis.csv', index = False)
    print(f"✅ Dataframe with {match_data.shape[0]} boards and {match_data.shape[1]} columns saved to analysis.csv.")

    # Pipeline for optimiser. Only retains 1st place boards
    pipe_opt = Pipeline([
        ("dropnonfirst", DropNonFirst()),
        ("removetftprefix", RemoveTFTPrefix()),
    ])
    match_data = pipe_opt.fit_transform(match_data)

    train_opt, val_opt = generate_unit_training_pairs(match_data)

    train_opt.to_csv('data/train_opt.csv', index = False)
    val_opt.to_csv('data/val_opt.csv', index = False)
    print(f"✅ Dataframe with {train_opt.shape[0]} rows and {train_opt.shape[1]} columns saved to train_opt.csv.")
    print(f"✅ Dataframes with {val_opt.shape[0]} rows and {val_opt.shape[1]} columns saved to val_opt.csv.")
    #save_to_sql(match_data, "opt", sql_server, sql_database, username, password)              # Azure is expensive!

def save_to_sql(df, table_name, server, database, username, password, if_exists='replace'):
    try:
        connection_string = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};DATABASE={database};UID={username};PWD={password}"
        )

        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")

        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        print(f"✅ DataFrame with {df.shape[0]} boards and {df.shape[1]} columns saved to table "
              f"'{table_name}' in Azure SQL database '{database}'.")

    except Exception as e:
        print(f"❌ Error saving DataFrame to Azure SQL: {e}")

#--------------------------------------
# Run ETL
#--------------------------------------

if __name__ == '__main__':
    load_dotenv('.env')
    api_key = os.getenv('RIOT_API_KEY')

    dbs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(pipeline_for_server, platform, api_key) for platform in platforms]
        for future in concurrent.futures.as_completed(futures):
            db = future.result()
            if not db.empty:
                dbs.append(db)

    dbs = pd.concat(dbs, ignore_index=True)
    use_data_pipeline(dbs)