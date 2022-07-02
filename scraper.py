import re
import threading
from _csv import writer
from queue import Queue
from urllib.request import urlopen

import numpy as np
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
base_url = 'https://www.legaseriea.it/it/serie-a/'
match_cols = ['season', 'round'] + \
             ['date', 'time', 'referee', 'home_team', 'away_team', 'home_score', 'away_score'] + \
             ['home_coach'] + \
             ['home_player_' + str(i) for i in range(1, 12)] + \
             ['home_substitute_' + str(i) for i in range(1, 13)] + \
             ['away_coach'] + \
             ['away_player_' + str(i) for i in range(1, 12)] + \
             ['away_substitute_' + str(i) for i in range(1, 13)]


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


def scrape_match_referee(report_parent_div):
    if report_parent_div.getText().__contains__('Stadio:'):
        referee_firstname = report_parent_div.getText().split(':')[3].split(' ')[1]
        referee_lastname = report_parent_div.getText().split(':')[3].split(' ')[2]
    else:
        referee_firstname = report_parent_div.getText().split(':')[2].split(' ')[1]
        referee_lastname = report_parent_div.getText().split(':')[2].split(' ')[2]
    return referee_firstname + ' ' + referee_lastname


def was_match_won_by_forfeit(report_parent_div):
    return report_parent_div.getText().__contains__('Partita sospesa') | \
           report_parent_div.getText().__contains__('Partita non disputata')


def scrape_match_report(bs: BeautifulSoup):
    home_team = bs.find(class_='report-squadra squadra-a').getText()
    away_team = bs.find(class_='report-squadra squadra-b').getText()
    print("Scraping {} vs {}...".format(home_team, away_team))
    report_parent_div = bs.find(class_='report-data')
    if was_match_won_by_forfeit(report_parent_div):
        return ['<sus>' for _ in range(1, 8)]
    datetime = report_parent_div.findChild(name='span').getText()
    date = datetime.split(' - ')[0]
    time = datetime.split(' - ')[1]
    referee = scrape_match_referee(report_parent_div)
    home_team_score = bs.find(class_='squadra-risultato squadra-a').getText()
    away_team_score = bs.find(class_='squadra-risultato squadra-b').getText()
    return [date, time, referee, home_team, away_team, home_team_score, away_team_score]


def read_player_name_inside_row_tag(player_row_tag):
    index = 0
    for el in player_row_tag.find_all(name='td'):
        if index == 1:
            return el.getText()
        index += 1


def scrape_players_from_table(table_tag) -> list[str]:
    players = []
    player_row_tags = table_tag.find(name='tbody').find_all(name='tr')
    for player_row_tag in player_row_tags:
        dirty_player = read_player_name_inside_row_tag(player_row_tag)
        player = re.sub(r'[^a-zA-Z ]', '', dirty_player)
        players.append(player)
    return players


def scrape_match_home_team_coach(bs: BeautifulSoup) -> list[str]:
    home_team_coach_table_tag = bs.find_all(class_='tabella')[4]
    return scrape_players_from_table(home_team_coach_table_tag)


def scrape_match_home_team_on_pitch(bs: BeautifulSoup) -> list[str]:
    home_team_on_pitch_table_tag = bs.find_all(class_='tabella')[0]
    return scrape_players_from_table(home_team_on_pitch_table_tag)


def scrape_match_home_team_substitutes(bs: BeautifulSoup) -> list[str]:
    home_team_substitutes_table_tag = bs.find_all(class_='tabella')[2]
    return scrape_players_from_table(home_team_substitutes_table_tag)


def scrape_match_home_team_lineup(bs: BeautifulSoup) -> list[str]:
    home_team_coach = scrape_match_home_team_coach(bs)
    home_team_on_pitch = scrape_match_home_team_on_pitch(bs)
    home_team_substitutes = scrape_match_home_team_substitutes(bs)
    # starting from season 2012-13, the number of substitutes is inconsistent
    home_team_substitutes += ['-' for _ in range(12 - len(home_team_substitutes))]
    return home_team_coach + home_team_on_pitch + home_team_substitutes


def scrape_match_away_team_coach(bs: BeautifulSoup) -> list[str]:
    away_team_coach_table_tag = bs.find_all(class_='tabella')[5]
    return scrape_players_from_table(away_team_coach_table_tag)


def scrape_match_away_team_on_pitch(bs: BeautifulSoup) -> list[str]:
    away_team_on_pitch_table_tag = bs.find_all(class_='tabella')[1]
    return scrape_players_from_table(away_team_on_pitch_table_tag)


def scrape_match_away_team_substitutes(bs: BeautifulSoup) -> list[str]:
    away_team_substitutes_table_tag = bs.find_all(class_='tabella')[3]
    return scrape_players_from_table(away_team_substitutes_table_tag)


def scrape_match_away_team_lineup(bs: BeautifulSoup) -> list[str]:
    away_team_coach = scrape_match_away_team_coach(bs)
    away_team_on_pitch = scrape_match_away_team_on_pitch(bs)
    away_team_substitutes = scrape_match_away_team_substitutes(bs)
    # starting from season 2012-13, the number of substitutes is inconsistent
    away_team_substitutes += ['-' for _ in range(12 - len(away_team_substitutes))]
    return away_team_coach + away_team_on_pitch + away_team_substitutes


def scrape_match_stats(bs: BeautifulSoup) -> list[str]:
    stats_parent_div = bs.find(class_='numeridelmatch')
    dict = {}
    for row in stats_parent_div.findChildren(class_='riga'):
        dict[row.findChild(class_='valoretitolo').getText()] = \
            [row.findChild(class_='valoresx').getText(), row.findChild(class_='valoredx').getText()]
    stats = dict['Parate'] + dict['Rigori'] + dict['Tiri Totali'] + dict['Tiri in porta'] + \
            dict['Tiri fuori'] + dict['Tiri in porta da area'] + dict['Falli commessi'] + \
            dict['Pali'] + dict['Assist Totali'] + dict['Fuorigioco'] + dict['Corner'] + \
            dict['Ammonizioni'] + dict['Espulsioni'] + dict['Cross'] + dict['Lanci lunghi'] + \
            dict['Attacchi centrali'] + dict['Attacchi a destra'] + dict['Attacchi a sinistra']
    return stats


def scrape_match_team_lineups(bs: BeautifulSoup) -> list[str]:
    home_team = scrape_match_home_team_lineup(bs)
    away_team = scrape_match_away_team_lineup(bs)
    return home_team + away_team


def scrape_match_data(bs: BeautifulSoup) -> list[str]:
    match_report = scrape_match_report(bs)
    match_stats = scrape_match_stats(bs)
    match_teams = scrape_match_team_lineups(bs)
    return match_report + match_stats + match_teams


def fetch_and_parse_and_queue_match_page(match_url, queue):
    url = base_url + match_url
    html = urlopen(url)
    bs = BeautifulSoup(html.read(), 'html.parser')
    queue.put(bs)


def fetch_all_matches_pages_async(matches_uris, queue):
    threads = [threading.Thread(target=fetch_and_parse_and_queue_match_page, args=(url, queue)) for url in matches_uris]
    for t in threads:
        t.start()


def scrape():
    years = np.arange(2005, 2021, 1)
    seasons = np.array(["{}-{}".format(years[i], years[i] + 1).replace('-20', '-') for i in range(years.size)])
    exp_num_of_matches_per_round = 10
    print(f'Will scrape data from season {seasons[0]} to season {seasons[-1]}')
    rounds = np.arange(1, 39, 1)
    csv = open('raw.csv', 'a', newline='')
    write_obj = writer(csv)
    write_obj.writerow(match_cols)
    for season in seasons:
        for round in rounds:
            print("Season {} Round {}:".format(season, round))
            matches_uris = scrape_round_matches_urls(season, round)
            queue = Queue()
            fetch_all_matches_pages_async(matches_uris, queue)
            scraped_count = 0
            while scraped_count < len(matches_uris):
                match_data = scrape_match_data(queue.get())
                scraped_count += 1
                write_obj.writerow([season, round] + match_data)
            if len(matches_uris) < exp_num_of_matches_per_round:
                print(f'WARNING: Only {len(matches_uris)} matches were found in round {round} of season {season}')
                for _ in range(exp_num_of_matches_per_round - len(matches_uris)):
                    write_obj.writerow([season, round] + ['-' for _ in range(len(match_cols) - 2)])
    csv.close()


if __name__ == '__main__':
    scrape()
