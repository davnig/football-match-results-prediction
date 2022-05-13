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
             ['home_player_' + str(i) for i in range(1, 12)] + \
             ['away_player_' + str(i) for i in range(1, 12)]


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
    print(urls)
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


def read_player_name_inside_parent_tag(player_parent_tag):
    index = 0
    for el in player_parent_tag.find_all(name='td'):
        if index == 1:
            return el.getText()
        index += 1


def scrape_match_on_pitch_team(on_pitch_team_parent_tag):
    players = []
    on_pitch_players_parent_tags = on_pitch_team_parent_tag.find(name='tbody').find_all(name='tr')
    for on_pitch_player_parent_tag in on_pitch_players_parent_tags:
        dirty_player = read_player_name_inside_parent_tag(on_pitch_player_parent_tag)
        player = re.sub(r'[^a-zA-Z ]', '', dirty_player)
        players.append(player)
    return players


def scrape_match_on_pitch_home_team(bs: BeautifulSoup):
    players = []
    on_pitch_home_team_parent_tag = bs.find(class_='colonna-squadra')
    return scrape_match_on_pitch_team(on_pitch_home_team_parent_tag)


def scrape_match_substitutes_home_team(bs: BeautifulSoup):
    # todo
    return []


def scrape_match_home_team(bs: BeautifulSoup):
    on_pitch_home_team = scrape_match_on_pitch_home_team(bs)
    substitutes_home_team = scrape_match_substitutes_home_team(bs)
    return on_pitch_home_team + substitutes_home_team


def scrape_match_on_pitch_away_team(bs: BeautifulSoup):
    away_team = []
    away_team_parent_tag = bs.find(class_='colonna-squadra').find_next(class_='colonna-squadra')
    return scrape_match_on_pitch_team(away_team_parent_tag)


def scrape_match_substitutes_away_team(bs: BeautifulSoup):
    # todo
    return []


def scrape_match_away_team(bs: BeautifulSoup):
    on_pitch_away_team = scrape_match_on_pitch_away_team(bs)
    substitutes_home_team = scrape_match_substitutes_away_team(bs)
    return on_pitch_away_team + substitutes_home_team


def scrape_match_teams(bs: BeautifulSoup):
    home_team = scrape_match_home_team(bs)
    away_team = scrape_match_away_team(bs)
    return home_team + away_team


def scrape_match_data(match_url):
    url = base_url + match_url
    html = urlopen(url)
    bs = BeautifulSoup(html.read(), 'html.parser')
    match_report = scrape_match_report(bs)
    match_teams = scrape_match_teams(bs)
    return match_report + match_teams


def scrape_data() -> pd.DataFrame:
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
            for match_url in matches_uris:
                match_data = pd.DataFrame([scrape_match_data(match_url)], columns=match_cols)
                seasons_data = pd.concat([seasons_data, match_data], ignore_index=True)
    browser.quit()
    return seasons_data
