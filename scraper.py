import re
from urllib.request import urlopen

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
base_url = 'https://www.legaseriea.it/it/serie-a/'
match_cols = ['date', 'time', 'referee', 'home_team', 'away_team', 'home_team_score', 'away_team_score'] + \
             ['home_team_coach'] + \
             ['home_player_' + str(i) for i in range(1, 12)] + \
             ['home_substitute_' + str(i) for i in range(1, 8)] + \
             ['away_team_coach'] + \
             ['away_player_' + str(i) for i in range(1, 12)] + \
             ['away_substitute_' + str(i) for i in range(1, 8)]


def init_headless_browser():
    chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--start-maximised")
    return webdriver.Chrome(
        executable_path="D:\\Windows\\Programmi\\phantomjs-2.1.1\\chromedriver_win32\\chromedriver.exe",
        options=chrome_options)


def scrape_round_matches_urls(season, round):
    season_and_round_url_part = '{}/UNICO/UNI/{}'.format(season, round)
    url = base_url + 'archivio/' + season_and_round_url_part
    html = urlopen(url)
    bs = BeautifulSoup(html.read(), 'html.parser')
    matches_a_tags = bs.find_all('a', {
        'href': lambda x: x and x.startswith('/it/serie-a/match-report/' + season_and_round_url_part)})
    urls = []
    for tag in matches_a_tags:
        urls.append(tag['href'].replace('/it/serie-a/', ''))
    return urls


def scrape_match_report(bs: BeautifulSoup):
    report_parent_div = bs.find(class_='report-data')
    datetime = report_parent_div.findChild(name='span').getText()
    date = datetime.split(' - ')[0]
    time = datetime.split(' - ')[1]
    referee_firstname = report_parent_div.getText().split(':')[3].split(' ')[1]
    referee_lastname = report_parent_div.getText().split(':')[3].split(' ')[2]
    referee = referee_firstname + ' ' + referee_lastname
    home_team = bs.find(class_='report-squadra squadra-a').getText()
    home_team_score = bs.find(class_='squadra-risultato squadra-a').getText()
    away_team = bs.find(class_='report-squadra squadra-b').getText()
    away_team_score = bs.find(class_='squadra-risultato squadra-b').getText()
    return [date, time, referee, home_team, away_team, home_team_score, away_team_score]


def read_player_name_inside_row_tag(player_row_tag):
    index = 0
    for el in player_row_tag.find_all(name='td'):
        if index == 1:
            return el.getText()
        index += 1


def scrape_players_from_table(table_tag):
    players = []
    player_row_tags = table_tag.find(name='tbody').find_all(name='tr')
    for player_row_tag in player_row_tags:
        dirty_player = read_player_name_inside_row_tag(player_row_tag)
        player = re.sub(r'[^a-zA-Z ]', '', dirty_player)
        players.append(player)
    return players


def scrape_match_home_team_coach(bs: BeautifulSoup):
    home_team_coach_table_tag = bs.find_all(class_='tabella')[4]
    return scrape_players_from_table(home_team_coach_table_tag)


def scrape_match_home_team_on_pitch(bs: BeautifulSoup):
    home_team_on_pitch_table_tag = bs.find_all(class_='tabella')[0]
    return scrape_players_from_table(home_team_on_pitch_table_tag)


def scrape_match_home_team_substitutes(bs: BeautifulSoup):
    home_team_substitutes_table_tag = bs.find_all(class_='tabella')[2]
    return scrape_players_from_table(home_team_substitutes_table_tag)


def scrape_match_home_team_lineup(bs: BeautifulSoup):
    home_team_coach = scrape_match_home_team_coach(bs)
    home_team_on_pitch = scrape_match_home_team_on_pitch(bs)
    home_team_substitutes = scrape_match_home_team_substitutes(bs)
    return home_team_coach + home_team_on_pitch + home_team_substitutes


def scrape_match_away_team_coach(bs: BeautifulSoup):
    away_team_coach_table_tag = bs.find_all(class_='tabella')[5]
    return scrape_players_from_table(away_team_coach_table_tag)


def scrape_match_away_team_on_pitch(bs: BeautifulSoup):
    away_team_on_pitch_table_tag = bs.find_all(class_='tabella')[1]
    return scrape_players_from_table(away_team_on_pitch_table_tag)


def scrape_match_away_team_substitutes(bs: BeautifulSoup):
    away_team_substitutes_table_tag = bs.find_all(class_='tabella')[3]
    return scrape_players_from_table(away_team_substitutes_table_tag)


def scrape_match_away_team_lineup(bs: BeautifulSoup):
    away_team_coach = scrape_match_away_team_coach(bs)
    away_team_on_pitch = scrape_match_away_team_on_pitch(bs)
    away_team_substitutes = scrape_match_away_team_substitutes(bs)
    return away_team_coach + away_team_on_pitch + away_team_substitutes


def scrape_match_team_lineups(bs: BeautifulSoup):
    home_team = scrape_match_home_team_lineup(bs)
    away_team = scrape_match_away_team_lineup(bs)
    return home_team + away_team


def scrape_match_data(match_url):
    url = base_url + match_url
    html = urlopen(url)
    bs = BeautifulSoup(html.read(), 'html.parser')
    match_report = scrape_match_report(bs)
    match_teams = scrape_match_team_lineups(bs)
    return match_report + match_teams


def scrape() -> pd.DataFrame:
    browser = init_headless_browser()
    years = np.arange(2005, 2006, 1)
    seasons = np.array(["{}-{}".format(years[i], years[i] + 1).replace('-20', '-') for i in range(years.size)])
    rounds = np.arange(1, 39, 1)
    seasons_data = pd.DataFrame(columns=match_cols)
    for season in seasons:
        print("Scraping season {}...".format(season))
        for round in rounds:
            print("Round {}".format(round))
            matches_uris = scrape_round_matches_urls(season, round)
            for i in range(len(matches_uris)):
                print("Match nr. {}".format(i + 1))
                match_url = matches_uris[i]
                match_data = scrape_match_data(match_url)
                match_data = pd.DataFrame([match_data], columns=match_cols)
                seasons_data = pd.concat([seasons_data, match_data], ignore_index=True)
    browser.quit()
    return seasons_data
