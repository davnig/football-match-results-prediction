import pandas as pd

from utils import LINE_FLUSH


def get_column_names_containing_str(df: pd.DataFrame, substring: str) -> list[str]:
    return df.loc[:, df.columns.str.contains(substring)].columns.values.tolist()


def encode_seasons(df: pd.DataFrame):
    print('Encoding seasons...', end=' ')
    season2index = {'20' + f'{i + 5}'.zfill(2) + '-' + f'{i + 6}'.zfill(2): i for i in range(16)}
    for col in df.filter(regex='season', axis=1).columns:
        df[col] = df[col].map(season2index)
    print(f'DONE. Final shape: {df.shape}')
    return df


def encode_players(df: pd.DataFrame):
    """We need to manually one-hot encode the names of the players because we have to consider all the player columns
        to construct a list containing all the unique values. Moreover, the number of substitutes is inconsistent across
        seasons, rounds and matches."""
    print('Encoding players...', end=' ')
    player_columns = df.filter(regex='player|substitute', axis=1).columns.tolist()
    home_player_columns = df.filter(regex='home_player|home_substitute', axis=1).columns.tolist()
    # home_player_columns = get_column_names_containing_str(df, 'home_player')
    # home_player_columns += get_column_names_containing_str(df, 'home_substitute')
    away_player_columns = df.filter(regex='away_player|away_substitute', axis=1).columns.tolist()
    # away_player_columns = get_column_names_containing_str(df, 'away_player')
    # away_player_columns += get_column_names_containing_str(df, 'away_substitute')
    all_unique_players = pd.concat([df[player_columns[i]] for i in range(len(player_columns))],
                                   axis=0).unique().tolist()
    all_unique_players.remove('-')
    result = []
    for index, row in df.iterrows():
        percentage = int((index + 1) * 100.0 / len(df))
        print(f'{LINE_FLUSH}Encoding players... {percentage} %', end=' ')
        home_encoding = [0] * len(all_unique_players)
        away_encoding = [0] * len(all_unique_players)
        for idx in range(len(home_player_columns)):
            if row[home_player_columns[idx]] != '-':
                home_player_idx = all_unique_players.index(row[home_player_columns[idx]])
                home_encoding[home_player_idx] = 1
            if row[away_player_columns[idx]] != '-':
                away_player_idx = all_unique_players.index(row[away_player_columns[idx]])
                away_encoding[away_player_idx] = 1
        result += [home_encoding + away_encoding]
    home_away_player_columns = [f'home_player_{name}' for name in all_unique_players] + [f'away_player_{name}' for
                                                                                         name in all_unique_players]
    players_df = pd.DataFrame(result, columns=home_away_player_columns)
    df = df.drop(player_columns, axis=1)
    df = pd.concat([df, players_df.set_index(df.index)], axis=1)
    print(f'{LINE_FLUSH}Encoding players... DONE. Final shape: {df.shape}')
    return df


def remove_players(df: pd.DataFrame):
    return df.drop(df.filter(regex='player|substitute', axis=1).columns, axis=1)


def encode_remaining_feats(df: pd.DataFrame):
    df = pd.get_dummies(df, dtype=int)
    print(f'Remaining feats encoded. Shape: {df.shape}')
    return df


def shift_home_team_cols_to_end(df: pd.DataFrame):
    cols_to_shift = get_column_names_containing_str(df, 'home_team')
    players_df = df[cols_to_shift].copy()
    df = df.drop(cols_to_shift, axis=1)
    df = pd.concat([df, players_df], axis=1)
    return df


def shift_away_team_cols_to_end(df: pd.DataFrame):
    cols_to_shift = get_column_names_containing_str(df, 'away_team')
    players_df = df[cols_to_shift].copy()
    df = df.drop(cols_to_shift, axis=1)
    df = pd.concat([df, players_df], axis=1)
    return df


def shift_referee_cols_to_end(df: pd.DataFrame):
    cols_to_shift = get_column_names_containing_str(df, 'referee')
    players_df = df[cols_to_shift].copy()
    df = df.drop(cols_to_shift, axis=1)
    df = pd.concat([df, players_df], axis=1)
    return df


def shift_player_cols_to_end(df: pd.DataFrame):
    cols_to_shift = get_column_names_containing_str(df, 'player')
    players_df = df[cols_to_shift].copy()
    df = df.drop(cols_to_shift, axis=1)
    df = pd.concat([df, players_df], axis=1)
    return df