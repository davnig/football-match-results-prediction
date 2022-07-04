import pandas as pd
import torch

from _MatchNotFoundException import MatchNotFoundException

match_report_cols = ['season', 'round', 'date', 'time', 'referee', 'home_team', 'away_team', 'home_score', 'away_score']

match_stats_cols = ['home_gk_saves', 'away_gk_saves', 'home_penalties', 'away_penalties', 'home_shots', 'away_shots',
                    'home_shots_on_target', 'away_shots_on_target', 'home_shots_off_target', 'away_shots_off_target',
                    'home_shots_on_target_from_penalty_area', 'away_shots_on_target_from_penalty_area', 'home_fouls',
                    'away_fouls', 'home_woodwork_hits', 'away_woodwork_hits', 'home_goal_chances', 'away_goal_chances',
                    'home_assists', 'away_assists', 'home_offsides', 'away_offsides', 'home_corner_kicks',
                    'away_corner_kicks', 'home_yel_cards', 'away_yel_cards', 'home_red_cards', 'away_red_cards',
                    'home_crosses', 'away_crosses', 'home_long_throws', 'away_long_throws', 'home_attacks_from_center',
                    'away_attacks_from_center', 'home_attacks_from_right', 'away_attacks_from_right',
                    'home_attacks_from_left', 'away_attacks_from_left']

match_teams_cols = ['home_coach'] + \
                   ['home_player_' + str(i) for i in range(1, 12)] + \
                   ['home_substitute_' + str(i) for i in range(1, 13)] + \
                   ['away_coach'] + \
                   ['away_player_' + str(i) for i in range(1, 12)] + \
                   ['away_substitute_' + str(i) for i in range(1, 13)]

match_cols = match_report_cols + match_stats_cols + match_teams_cols

"""Utility methods"""


def scale_idx(idx, old_min, old_max, new_min, new_max):
    """Scale the given index to a new range"""
    old_range = old_max - old_min
    new_range = new_max - new_min
    normalized_idx = (idx - old_min) / old_range
    return int(round(normalized_idx * new_range + new_min))


def accuracy(y, y_hat):
    _, pred_classes = torch.topk(y_hat, 1)
    _, target_classes = torch.topk(y, 1)
    correct = (pred_classes == target_classes).sum(dim=0)
    return (correct / y.shape[0]).item()


"""Historic matches retrieval"""


def get_match_by_id(df: pd.DataFrame, match_id: str) -> pd.DataFrame:
    return df[df['match_id'] == match_id]


def get_match_index_by_id(df: pd.DataFrame, match_id: str) -> int:
    return get_match_by_id(df, match_id).index.tolist()[0]


def get_match_index_by_match(match: pd.DataFrame) -> int:
    return match.index.tolist()[0]


def is_team_home_or_away_in_match(team_name: str, match: pd.DataFrame):
    home_team = match.squeeze()['home_team']
    if home_team == team_name:
        return 'home'
    else:
        return 'away'


def get_last_match_played_by_team(df: pd.DataFrame, target_match_index: int, team_name: str) -> pd.DataFrame:
    """
    Find in df the last match played by team_name prior to the game identified by target_match_index
    :param df: where to search
    :param target_match_index: the index in df of the target match
    :param team_name: name of the team that has played the target match
    :return:
    """
    for i in reversed(range(target_match_index)):
        current_match = df.iloc[[i]]  # dataframe
        if current_match.at[i, 'home_team'] == team_name or current_match.at[i, 'away_team'] == team_name:
            return current_match
    raise MatchNotFoundException(f'Previous match for team {team_name} was not found')


def get_last_five_matches_played_by_team(df: pd.DataFrame, target_match_index: int, team_name: str) -> list[pd.Series]:
    """
    Find in df the last five matches played by team_name prior to the game identified by match_index
    :param df: where to search
    :param target_match_index: target match index in df
    :param team_name: the name of the team that has played all the last five matches
    :return:
    """
    match = df.iloc[[target_match_index]]  # dataframe
    last_match_found = match
    last_five_matches = []
    for i in range(1, 6):
        try:
            last_match_found = get_last_match_played_by_team(df, get_match_index_by_match(last_match_found), team_name)
            last_five_matches.append(last_match_found.squeeze())
        except MatchNotFoundException:
            pass
    return last_five_matches


def transform_historic_data_long_to_wide(long_df: pd.DataFrame, target_home_or_away: str, target_idx: int,
                                         hist_len: int) -> pd.DataFrame:
    # Init columns for n historic matches
    historic_cols = [f'{target_home_or_away}_hist_{i}_{colName}' for i in range(1, hist_len + 1) for colName in
                     match_cols]
    # Init empty DataFrame with those columns and specific index
    result = pd.DataFrame(columns=historic_cols, index=[target_idx])
    # Copy values into DataFrame
    for i in range(len(long_df)):
        source_match = long_df[i]
        for colName, colValue in source_match.iteritems():
            result.at[target_idx, f'{target_home_or_away}_hist_{i + 1}_{colName}'] = colValue
    return result


def get_last_n_matches_played_by_home_and_away_teams(df: pd.DataFrame, season: int, round: int, home_team: str,
                                                     away_team: str, history_len=5):
    """Retrieve from df the historical data for the given home and away teams prior to the given round in season."""

    def get_last_n_matches_played_by_team_before_round_in_season(df: pd.DataFrame, team: str, season: int,
                                                                 round: int, n: int) -> pd.DataFrame:
        """Look in df for the last n matches played by the given team before the given round and in the given season.
        A dataframe with exactly n element is returned. If n matches can't be found in the given season, padding
        is applied to ensure a result size of n."""

        def fill_with_padding(source: pd.DataFrame):
            if len(source) < 5:
                padding = source.tail(1)
                for i in range(5 - len(source)):
                    source = pd.concat([source, padding], ignore_index=True)
            return source

        def get_match_by_team_season_round(df: pd.DataFrame, team: str, season: int, round: int) -> pd.DataFrame:
            """Get the match played by the given team in the given season and round.
            If the team has not played any match in that round, an empty dataframe is returned."""
            return df[((df[f'home_team'] == team) | (df[f'away_team'] == team)) & (df['round'] == round) & (
                    df['season'] == season)]

        current_round, current_season = round, season
        result = pd.DataFrame()
        while True:
            if round <= 1:
                if result.empty:
                    raise MatchNotFoundException
                return fill_with_padding(result)
            current_round = current_round - 1
            historical_match_at_current_round = get_match_by_team_season_round(df, team, current_season, current_round)
            if not historical_match_at_current_round.empty:
                result = pd.concat([result, historical_match_at_current_round])
                if len(result) == n:
                    return result

    last_n_games_home = get_last_n_matches_played_by_team_before_round_in_season(
        df, home_team, season, round, history_len)
    last_n_games_away = get_last_n_matches_played_by_team_before_round_in_season(
        df, away_team, season, round, history_len)
    return last_n_games_home, last_n_games_away


def add_historic_data_of_last_n_matches_as_features(df: pd.DataFrame, history_len=5) -> pd.DataFrame:
    """
    Construct and return a new dataframe adding information about the last five matches played by home and away team of all
    matches in df.
    :param df: source of data
    :return: a new dataframe
    """
    new_df = pd.DataFrame()
    # for each row in dataframe
    for (index, season, round, home_team, away_team) in df.itertuples(name=None):
        home_historic_df, away_historic_df = get_last_n_matches_played_by_home_and_away_teams(df, season,
                                                                                              round,
                                                                                              home_team,
                                                                                              away_team,
                                                                                              history_len)
        wide_home_historic_df = transform_historic_data_long_to_wide(home_historic_df, 'home', index, history_len)
        wide_away_historic_df = transform_historic_data_long_to_wide(away_historic_df, 'away', index, history_len)
        new_row_as_df = pd.concat([df.iloc[[index]], wide_home_historic_df, wide_away_historic_df], axis=1)
        new_df = pd.concat([new_df, new_row_as_df], axis=0)
    return new_df
