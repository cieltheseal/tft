import requests
import pandas as pd
import time

from flatten_json import flatten

#-------------------------------------------
# Get PUUIDs of challengers from a platform
#-------------------------------------------

def get_challengers(platform, api_key) -> list:
    puuids = []
    get_challengers_url = ('https://' + platform + '.api.riotgames.com/tft/league/v1/challenger?api_key=' + api_key)
    print(f"Extracting Challenger PUUIDs from {platform}.")

    try:
        challengers_resp = requests.get(get_challengers_url, timeout=60)
        challengers_resp.raise_for_status()
        challengers_info = challengers_resp.json()
        puuids.extend(entry['puuid'] for entry in challengers_info['entries'])
        print(f"{platform} Challenger PUUIDs successfully extracted.")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return puuids

#----------------------------------------------
# Get data on matches and convert to DataFrame
#----------------------------------------------

def get_matches(puuids, region, n, api_key) -> list:
    print(f"Extracting {n} most recent matches for {len(puuids)} Challengers from {region}.")

    match_ids = set()
    for puuid in puuids:
        matches_url = ('https://' + region + '.api.riotgames.com/tft/match/v1/matches/by-puuid/'
                       + puuid + '/ids?start=0&count=' + str(n) + '&api_key=' + api_key)

        try:
            matches_resp = requests.get(matches_url, timeout = 60)
            matches_resp.raise_for_status()
            match_ids.update(matches_resp.json())

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    print(f"{len(match_ids)} Match IDs successfully extracted.")
    return match_ids

def get_match_info(match_id, region, api_key) -> dict:
    match_url = ('https://' + region + '.api.riotgames.com/tft/match/v1/matches/' + match_id + '?api_key=' + api_key)
    try:
        match_resp = requests.get(match_url, timeout = 60)
        match_resp.raise_for_status()
        match_info = match_resp.json()
        return match_info["info"]
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def get_matches_info(matches, region, api_key) -> pd.DataFrame():
    match_data = pd.DataFrame()
    print("Obtaining match data...")

    for match in matches:
        time.sleep(1)
        match_info = get_match_info(match, region, api_key)
        queue_id = match_info.get('queue_id')
        if queue_id != 1090:               # Only select ranked games
            continue
        flat_match_resp = flatten(match_info)
        keyList = list(flat_match_resp.keys())

        keyList1 = [key for key in keyList if key.startswith('participants_0')]
        keyList2 = [key for key in keyList if key.startswith('participants_1')]
        keyList3 = [key for key in keyList if key.startswith('participants_2')]
        keyList4 = [key for key in keyList if key.startswith('participants_3')]
        keyList5 = [key for key in keyList if key.startswith('participants_4')]
        keyList6 = [key for key in keyList if key.startswith('participants_5')]
        keyList7 = [key for key in keyList if key.startswith('participants_6')]
        keyList8 = [key for key in keyList if key.startswith('participants_7')]

        player1info = {k:v for k, v in flat_match_resp.items() if k in keyList1}
        player2info = {k:v for k, v in flat_match_resp.items() if k in keyList2}
        player3info = {k:v for k, v in flat_match_resp.items() if k in keyList3}
        player4info = {k:v for k, v in flat_match_resp.items() if k in keyList4}
        player5info = {k:v for k, v in flat_match_resp.items() if k in keyList5}
        player6info = {k:v for k, v in flat_match_resp.items() if k in keyList6}
        player7info = {k:v for k, v in flat_match_resp.items() if k in keyList7}
        player8info = {k:v for k, v in flat_match_resp.items() if k in keyList8}

        player1 = pd.DataFrame([player1info])
        player2 = pd.DataFrame([player2info])
        player3 = pd.DataFrame([player3info])
        player4 = pd.DataFrame([player4info])
        player5 = pd.DataFrame([player5info])
        player6 = pd.DataFrame([player6info])
        player7 = pd.DataFrame([player7info])
        player8 = pd.DataFrame([player8info])

        player1.columns = player1.columns.str.replace('participants_0_', '')
        player2.columns = player2.columns.str.replace('participants_1_', '')
        player3.columns = player3.columns.str.replace('participants_2_', '')
        player4.columns = player4.columns.str.replace('participants_3_', '')
        player5.columns = player5.columns.str.replace('participants_4_', '')
        player6.columns = player6.columns.str.replace('participants_5_', '')
        player7.columns = player7.columns.str.replace('participants_6_', '')
        player8.columns = player8.columns.str.replace('participants_7_', '')

        player1 = player1.convert_dtypes()
        player2 = player2.convert_dtypes()
        player3 = player3.convert_dtypes()
        player4 = player4.convert_dtypes()
        player5 = player5.convert_dtypes()
        player6 = player6.convert_dtypes()
        player7 = player7.convert_dtypes()
        player8 = player8.convert_dtypes()

        player1 = player1.select_dtypes(exclude=['object'])
        player2 = player2.select_dtypes(exclude=['object'])
        player3 = player3.select_dtypes(exclude=['object'])
        player4 = player4.select_dtypes(exclude=['object'])
        player5 = player5.select_dtypes(exclude=['object'])
        player6 = player6.select_dtypes(exclude=['object'])
        player7 = player7.select_dtypes(exclude=['object'])
        player8 = player8.select_dtypes(exclude=['object'])

        match_data = pd.concat([match_data, player1], ignore_index = True)
        match_data = pd.concat([match_data, player2], ignore_index = True)
        match_data = pd.concat([match_data, player3], ignore_index = True)
        match_data = pd.concat([match_data, player4], ignore_index = True)
        match_data = pd.concat([match_data, player5], ignore_index = True)
        match_data = pd.concat([match_data, player6], ignore_index = True)
        match_data = pd.concat([match_data, player7], ignore_index = True)
        match_data = pd.concat([match_data, player8], ignore_index = True)

    print(f"Dataframe with {match_data.shape[0]} boards and {match_data.shape[1]} columns generated.")

    return match_data
