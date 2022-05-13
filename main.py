import pandas as pd

import scraper


def save_in_csv(data: pd.DataFrame):
    data.to_csv('data.csv')


if __name__ == '__main__':
    scraper.scrape()
    # save_in_csv(seasons_data)
