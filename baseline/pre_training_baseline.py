import pandas as pd

if __name__ == '__main__':
    csv_name = 'data1.csv'
    df = pd.read_csv(csv_name)
    # let's delete the features that are not available pre-match
    df = df.drop(columns=['home_score', 'away_score'])
    df.to_csv("data_baseline.csv")
    print('DONE')
