import pandas as pd

import scrape as scraper


def save_in_csv(data: list[list[str]]):
    print(data)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv('data.csv')


if __name__ == '__main__':
    data = scraper.scrape_data()
    save_in_csv(data)
