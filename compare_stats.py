import os
import glob
import numpy as np
import pandas as pd


def compare_stats():
    # Remove old stats
    all_stats_files = glob.glob('*.csv')
    for stats_path in all_stats_files:
        try:
            os.remove(stats_path)
        except:
            pass

    # Get stats files of all submission agents
    all_stats_files = glob.glob('**/stats.csv', recursive=True)

    # Create a concatenated dataframe from all stats of all submission agents
    df_from_each_file = (pd.read_csv(f) for f in all_stats_files)
    concatenated_stats_df = pd.concat(df_from_each_file, ignore_index=True)

    # Create index list for rearrangement
    n_rows = concatenated_stats_df.index[-1] + 1
    indices = np.arange(n_rows)
    indices = list(sorted(indices, key=lambda x: [x % 2, x]))

    # Rearrange rows and save to .csv
    concatenated_stats_df = concatenated_stats_df.set_index(pd.Index(indices))
    concatenated_stats_df = concatenated_stats_df.sort_index()
    concatenated_stats_df.to_csv('all_stats.csv')

    # Split dataframe by agent names
    stats_info = concatenated_stats_df.sort_values(by=['Agent Name'])
    stats_info.reset_index(inplace=True, drop=True)
    stats_info_agent1 = stats_info.truncate(before=0, after=n_rows / 2 - 1)
    stats_info_agent2 = stats_info.truncate(before=n_rows / 2, after=n_rows - 1)

    # Get average of total city tiles for agents
    agent1_avg_city_tiles = round(stats_info_agent1['Total City Tiles'].mean(), 2)
    agent2_avg_city_tiles = round(stats_info_agent2['Total City Tiles'].mean(), 2)

    # Get average of total city tiles per map size for agents
    agent1_avg_city_tiles_12 = round(stats_info_agent1.loc[stats_info_agent1['Map Size'] == '12x12', 'Total City Tiles'].mean(), 2)
    agent1_avg_city_tiles_16 = round(stats_info_agent1.loc[stats_info_agent1['Map Size'] == '16x16', 'Total City Tiles'].mean(), 2)
    agent1_avg_city_tiles_24 = round(stats_info_agent1.loc[stats_info_agent1['Map Size'] == '24x24', 'Total City Tiles'].mean(), 2)
    agent1_avg_city_tiles_32 = round(stats_info_agent1.loc[stats_info_agent1['Map Size'] == '32x32', 'Total City Tiles'].mean(), 2)
    agent2_avg_city_tiles_12 = round(stats_info_agent2.loc[stats_info_agent2['Map Size'] == '12x12', 'Total City Tiles'].mean(), 2)
    agent2_avg_city_tiles_16 = round(stats_info_agent2.loc[stats_info_agent2['Map Size'] == '16x16', 'Total City Tiles'].mean(), 2)
    agent2_avg_city_tiles_24 = round(stats_info_agent2.loc[stats_info_agent2['Map Size'] == '24x24', 'Total City Tiles'].mean(), 2)
    agent2_avg_city_tiles_32 = round(stats_info_agent2.loc[stats_info_agent2['Map Size'] == '32x32', 'Total City Tiles'].mean(), 2)

    # Get average win rate for agents
    agent1_total_wins = stats_info_agent1['Result'].value_counts()
    agent1_total_wins = agent1_total_wins.get('WIN')
    agent2_total_wins = stats_info_agent2['Result'].value_counts()
    agent2_total_wins = agent2_total_wins.get('WIN')

    # Get average win rate per map size for agents
    agent1_total_wins_12 = stats_info_agent1.loc[stats_info_agent1['Map Size'] == '12x12', 'Result'].value_counts()
    agent1_total_wins_12 = agent1_total_wins_12.get('WIN')
    agent1_total_wins_16 = stats_info_agent1.loc[stats_info_agent1['Map Size'] == '16x16', 'Result'].value_counts()
    agent1_total_wins_16 = agent1_total_wins_16.get('WIN')
    agent1_total_wins_24 = stats_info_agent1.loc[stats_info_agent1['Map Size'] == '24x24', 'Result'].value_counts()
    agent1_total_wins_24 = agent1_total_wins_24.get('WIN')
    agent1_total_wins_32 = stats_info_agent1.loc[stats_info_agent1['Map Size'] == '32x32', 'Result'].value_counts()
    agent1_total_wins_32 = agent1_total_wins_32.get('WIN')
    agent2_total_wins_12 = stats_info_agent2.loc[stats_info_agent2['Map Size'] == '12x12', 'Result'].value_counts()
    agent2_total_wins_12 = agent2_total_wins_12.get('WIN')
    agent2_total_wins_16 = stats_info_agent2.loc[stats_info_agent2['Map Size'] == '16x16', 'Result'].value_counts()
    agent2_total_wins_16 = agent2_total_wins_16.get('WIN')
    agent2_total_wins_24 = stats_info_agent2.loc[stats_info_agent2['Map Size'] == '24x24', 'Result'].value_counts()
    agent2_total_wins_24 = agent2_total_wins_24.get('WIN')
    agent2_total_wins_32 = stats_info_agent2.loc[stats_info_agent2['Map Size'] == '32x32', 'Result'].value_counts()
    agent2_total_wins_32 = agent2_total_wins_32.get('WIN')

    # Get agent names
    agent_names = sorted(stats_info['Agent Name'].unique())

    # Create and save informational dataframe of stats as .csv
    stats_info_dict = {'Agent Name': agent_names, 'Average City Tiles': [agent1_avg_city_tiles, agent2_avg_city_tiles], 'Total Wins': [agent1_total_wins, agent2_total_wins]}
    stats_info_df = pd.DataFrame(data=stats_info_dict)
    stats_info_df.to_csv('stats_info.csv')

    # Create and save informational dataframe of stats per map size as .csv
    stats_info_per_map_dict = {'Agent Name': [agent_names[0], agent_names[1], agent_names[0], agent_names[1], agent_names[0], agent_names[1], agent_names[0], agent_names[1]], 'Average City Tiles': [agent1_avg_city_tiles_12, agent2_avg_city_tiles_12, agent1_avg_city_tiles_16, agent2_avg_city_tiles_16, agent1_avg_city_tiles_24, agent2_avg_city_tiles_24, agent1_avg_city_tiles_32, agent2_avg_city_tiles_32],'Total Wins': [agent1_total_wins_12, agent2_total_wins_12, agent1_total_wins_16, agent2_total_wins_16, agent1_total_wins_24, agent2_total_wins_24, agent1_total_wins_32, agent2_total_wins_32], 'Map Size': ['12x12', '12x12', '16x16', '16x16', '24x24', '24x24', '32x32', '32x32']}
    stats_info_per_map_df = pd.DataFrame(data=stats_info_per_map_dict)
    stats_info_per_map_df.to_csv('stats_info_per_map.csv')

    # Iterate over the list of all_stats_files and remove each file
    for stats_path in all_stats_files:
        try:
            os.remove(stats_path)
        except:
            pass
