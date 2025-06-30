from sklearn.pipeline import Pipeline
from extract import *
from transform import *

import urllib
import pyodbc
from sqlalchemy import create_engine

import concurrent.futures

api_key = 'RGAPI-blablabla'
platforms = ["NA1", "KR", "EUW1", "SG2"]
regions = {"Americas": ["NA1", "BR1"], "Asia": ["KR", "JP"], "Europe": ["EUW1"], "SEA": ["SG2"]}

sql_server = "tftopt.database.windows.net"
sql_database = "tftopt"

#------------------------------------------------
# Concurrent API calls from regional endpoints
#------------------------------------------------

def pipeline_for_server(platform, api_key, n_matches=10):
    try:
        print(f"🔄 Starting pipeline for {platform}")
        puuids = get_challengers(platform, api_key)
        region = next((region for region, platforms in regions.items() if platform in platforms), None)
        puuids = puuids[:5]
        match_ids = get_matches(puuids, region, n_matches, api_key)
        db = get_matches_info(match_ids, region, api_key)
        print(f"✅ Finished pipeline for {platform}")
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
    print(f"Dataframe with {match_data.shape[0]} boards and {match_data.shape[1]} columns saved to analysis.csv.")

    """
    # Pipeline for predictor
    pipe_analysis = Pipeline([
    ])

    match_data = pipe_analysis.fit_transform(match_data)
    match_data.to_csv('data/pred.csv', index = False)
    print(f"Dataframe with {match_data.shape[0]} boards and {match_data.shape[1]} columns saved.")
    """

    # Pipeline for optimiser. Only retains 1st place boards
    pipe_opt = Pipeline([
        ("dropnonfirst", DropNonFirst()),
        ("removetftprefix", RemoveTFTPrefix()),
    ])
    match_data = pipe_opt.fit_transform(match_data)

    train_opt, val_opt = generate_unit_training_pairs(match_data)

    train_opt.to_csv('data/train_opt.csv', index = False)
    val_opt.to_csv('data/val_opt.csv', index = False)
    print(f"Dataframes with {match_data.shape[0]} rows and {match_data.shape[1]} columns saved to _opt.csv.")
    #save_to_sql(match_data, "opt", sql_server, sql_database, username, password)              # Azure is expensive!

def save_to_sql(df, table_name, server, database, username, password, if_exists='replace'):
    """
    Saves a pandas DataFrame to an Azure SQL table.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        table_name (str): The target table name in Azure SQL.
        server (str): Azure SQL server name (e.g. 'your-server.database.windows.net').
        database (str): Name of the database on the server.
        username (str): SQL login username.
        password (str): SQL login password.
        if_exists (str): 'replace', 'append', or 'fail'. Default is 'replace'.

    Returns:
        None
    """
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
    dbs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(pipeline_for_server, platform, api_key) for platform in platforms]
        for future in concurrent.futures.as_completed(futures):
            db = future.result()
            if not db.empty:
                dbs.append(db)

    #puuids = get_challengers(platform, api_key)
    #region = next((region for region, platforms in regions.items() if platform in platforms), None)
    #puuids = puuids[0:10]
    #match_ids = get_matches(puuids, region, n, api_key)
    #db = get_matches_info(match_ids, region, api_key)
    dbs = pd.concat(dbs, ignore_index=True)
    use_data_pipeline(dbs)

"""
def get_profile(server, username, region, api_key) -> dict:
    profile_url = ('https://' + server + '.api.riotgames.com/riot/account/v1/accounts/by-riot-id/' + username
    + '/' + region+ '?api_key=' + api_key)
    try:
        profile_resp = requests.get(profile_url, timeout = 60)
        profile_info = profile_resp.json()
        print(profile_info)
        return profile_info
    except:
"""
"""
    # Pipeline for optimiser. Only retains 1st place boards
    pipe_opt = Pipeline([
        ("counttacticians", CountTacticianItems()),
        ("calculateboardsize", CalculateBoardSize()),
        ("dropnonfirst", DropNonFirst()),
        ("dropirrelevant", DropIrrelevant()),
        ("droptraits", DropTraits()),
        ("dropitems", DropItems()),
        ("droptierrarity", DropTierRarity()),
        ("dropskill", DropSkill()),
    ])
"""