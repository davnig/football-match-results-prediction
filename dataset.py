from random import randrange

import pandas as pd
import torch
from torch.utils.data import Dataset

from exceptions._MatchNotFoundException import MatchNotFoundException
from utils import scale_idx


class SerieAFootballMatchesDataset(Dataset):
    def __init__(self, csv_file, history_len=5):
        self.dataframe = pd.read_csv(csv_file)
        self.history_len = history_len

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        def show_error(index, error_x, error_x_historical_home, error_x_historical_away, error_y):
            print(f'error at index (scaled): {index} (unscaled): {self.unscale_min_idx(index)}')
            print(f'x: {error_x}')
            print(f'x.shape: {error_x.shape}')
            print(f'x_historical_home: {error_x_historical_home}')
            print(f'x_historical_home.shape: {error_x_historical_home.shape}')
            print(f'x_historical_away: {error_x_historical_away}')
            print(f'x_historical_away.shape: {error_x_historical_away.shape}')
            print(f'y: {error_y}')
            print(f'y.shape: {error_y.shape}')

        def to_tensor(x: pd.DataFrame, x_historical_home: pd.DataFrame, x_historical_away: pd.DataFrame,
                      y: list[int]):
            x_tensor = torch.flatten(torch.tensor(x.values))
            x_historical_home_tensor = torch.tensor(x_historical_home.values)
            x_historical_away_tensor = torch.tensor(x_historical_away.values)
            y_tensor = torch.tensor(y)
            return x_tensor, x_historical_home_tensor, x_historical_away_tensor, y_tensor

        idx = self.scale_min_idx(idx)
        x = self.dataframe.iloc[[idx]]  # df
        y = self.dataframe[['result_home', 'result_draw', 'result_away']].iloc[0].values
        try:  # if we are not able to fetch at least one historical match, then we switch to another index
            last_n_games_home, last_n_games_away = self.retrieve_historical_data(x)
            x, x_historical_home, x_historical_away, y = to_tensor(x, last_n_games_home, last_n_games_away, y)
            exp_num_of_features = len(self.dataframe.columns)
            if x.shape[0] != exp_num_of_features:
                show_error(idx, x, x_historical_home, x_historical_away, y)
            if (x_historical_home.shape[0] != 5) | (x_historical_home.shape[1] != exp_num_of_features):
                show_error(idx, x, x_historical_home, x_historical_away, y)
            if (x_historical_away.shape[0] != 5) | (x_historical_away.shape[1] != exp_num_of_features):
                show_error(idx, x, x_historical_home, x_historical_away, y)
            if y.shape[0] != 3:
                show_error(idx, x, x_historical_home, x_historical_away, y)
            return x, x_historical_home, x_historical_away, y
        except MatchNotFoundException:
            new_idx = randrange(0, len(self.dataframe))
            # print(f'MatchNotFoundException for idx={idx}, switching to idx={new_idx}')
            return self.__getitem__(new_idx)

    def scale_min_idx(self, idx: int) -> int:
        """Scale the given index to a range with a new minimum that allows for historical data retrieval"""
        old_min = 0
        old_max = len(self.dataframe)
        # idx = 10 corresponds to the first match of the second round.
        # This ensure the retrieval of at least 1 historical match.
        # In the worst case scenario, padding will fill the other 4 historical slots.
        new_min = 10
        new_max = old_max
        return scale_idx(idx, old_min, old_max, new_min, new_max)

    def unscale_min_idx(self, idx: int) -> int:
        """Apply the inverse transformation of scale_min_idx"""
        old_min = 10
        old_max = len(self.dataframe)
        new_min = 0
        new_max = old_max
        return scale_idx(idx, old_min, old_max, new_min, new_max)

    def retrieve_historical_data(self, source: pd.DataFrame):
        """Retrieve historical data for home and away teams from source"""

        def get_last_n_matches_played_by_team_before_round_in_season(df: pd.DataFrame, team: str, season: int,
                                                                     round: int, n: int) -> pd.DataFrame:
            """Look in df for the last n matches played by the given team before the given round and season.
            A dataframe with exactly n element is returned. If n matches can't be found from the current season,
            the previous ones are iteratively considered, until n matches are found or the end of the dataframe
            is reached, in which case padding is applied to ensure a result size of n."""

            def exists_historical_matches_before_round_and_season(q_round: int, q_season: int) -> bool:
                if (q_season == 0) & (q_round <= 1):
                    return False
                return True

            def decrement_round_in_season(c_round: int, c_season: int) -> (int, int):
                if c_round - 1 > 0:
                    c_round -= 1
                    return c_round, c_season
                c_season -= 1
                c_round = 38
                return c_round, c_season

            def fill_with_padding(source: pd.DataFrame):
                if len(source) < 5:
                    padding = source.tail(1)
                    for i in range(5 - len(source)):
                        source = pd.concat([source, padding], ignore_index=True)
                return source

            def get_match_by_team_season_round(df: pd.DataFrame, team: str, season: int, round: int) -> pd.DataFrame:
                """Get the match played by the given team in the given season and round.
                If the team has not played any match in that round, an empty dataframe is returned."""
                return df[((df[f'home_team_{team}'] == 1) | (df[f'away_team_{team}'] == 1)) & (df['round'] == round) & (
                        df['season'] == season)]

            current_round, current_season = round, season
            result = pd.DataFrame()
            while True:
                if not exists_historical_matches_before_round_and_season(current_round, current_season):
                    if result.empty:
                        raise MatchNotFoundException
                    return fill_with_padding(result)
                current_round, current_season = decrement_round_in_season(current_round, current_season)
                historical_match_at_current_round = get_match_by_team_season_round(df, team, current_season,
                                                                                   current_round)
                if not historical_match_at_current_round.empty:
                    result = pd.concat([result, historical_match_at_current_round])
                    if len(result) == n:
                        return result

        def get_playing_home_team_name(row: pd.DataFrame) -> str:
            team_columns = row.loc[:, [col for col in row.columns if col.startswith('home_team_')]]
            team_name = team_columns.where(team_columns == 1).dropna(axis=1).columns[0].replace('home_team_', '')
            return team_name

        def get_playing_away_team_name(row: pd.DataFrame) -> str:
            team_columns = row.loc[:, [col for col in row.columns if col.startswith('away_team_')]]
            team_name = team_columns.where(team_columns == 1).dropna(axis=1).columns[0].replace('away_team_', '')
            return team_name

        last_n_games_home = get_last_n_matches_played_by_team_before_round_in_season(
            self.dataframe, get_playing_home_team_name(source), source['season'].values[0], source['round'].values[0],
            self.history_len)
        last_n_games_away = get_last_n_matches_played_by_team_before_round_in_season(
            self.dataframe, get_playing_away_team_name(source), source['season'].values[0], source['round'].values[0],
            self.history_len)
        return last_n_games_home, last_n_games_away
