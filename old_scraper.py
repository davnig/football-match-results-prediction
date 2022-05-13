# import numpy as np
# import pandas as pd
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions
# from selenium.webdriver.support.wait import WebDriverWait
#
# chrome_options = Options()
# cols = ['date', 'time', 'team_A', 'team_B', 'goal_A', 'goal_B', 'fh_goal_A', 'fh_goal_B', 'match_id']
# cols1 = [
#     'round',
#     'datetime',
#     'home_team',
#     'away_team',
#     'home_score',
#     'away_score',
#     'fh_home_score',
#     'fh_away_score',
#     'home_player_1',
#     'home_player_2',
#     'home_player_3',
#     'home_player_4',
#     'home_player_5',
#     'home_player_6',
#     'home_player_7',
#     'home_player_8',
#     'home_player_9',
#     'home_player_10',
#     'home_player_11',
#     'home_player_12',
#     'home_player_13',
#     'home_player_14',
#     'away_player_1',
#     'away_player_2',
#     'away_player_3',
#     'away_player_4',
#     'away_player_5',
#     'away_player_6',
#     'away_player_7',
#     'away_player_8',
#     'away_player_9',
#     'away_player_10',
#     'away_player_11',
#     'away_player_12',
#     'away_player_13',
#     'away_player_14',
# ]
#
#
# def get_and_parse_season_page(browser: webdriver, season):
#     print('Parsing page for season {}...'.format(season))
#     url = "https://www.diretta.it/serie-a-{}/risultati".format(season)
#     browser.get(url)
#     # OPTIONAL: CLOSE BANNER
#     # try:
#     #     browser.find_element_by_id('onetrust-reject-all-handler').click()
#     # except ElementClickInterceptedException:
#     #     pass
#     for i in range(3):
#         show_more_btn = browser.find_element_by_css_selector('a.event__more')
#         browser.execute_script("arguments[0].scrollIntoView(true);", show_more_btn)
#         WebDriverWait(browser, 10).until(
#             expected_conditions.element_to_be_clickable((By.CSS_SELECTOR, 'a.event__more')))
#         show_more_btn.click()
#     html = browser.page_source
#     return BeautifulSoup(html, 'html.parser')
#
#
# def tokenize_match_data(match_data_parent_tag) -> list[str]:
#     match_data = ''
#     for child in match_data_parent_tag.findChildren():
#         match_data += child.getText() + ' '
#     return match_data.split()
#
#
# def handle_special_cases(match_data_list):
#     # Change 'ACR Messina' to 'Messina'
#     if 'ACR' in match_data_list:
#         match_data_list.remove('ACR')
#     return match_data_list
#
#
# def format_date(date, season):
#     month = int(date.split('.')[1])
#     if 8 <= month <= 12:
#         date = date + season.split('-')[0]
#     else:
#         date = date + season.split('-')[1]
#     return date
#
#
# def is_match_won_by_forfeit(match_data_list):
#     return 'A' in match_data_list and 'Tav.' in match_data_list
#
#
# def get_match_id_from_parent_tag(match_parent_tag):
#     return match_parent_tag.get_attribute_list('id')[0].replace('g_1_', '')
#
#
# # def get_match_teams_data(browser, bs, match_id) -> str:
# #     # Open a new tab
# #     browser.execute_script("window.open('');")
# #     # Switch to the new tab and open new URL
# #     browser.switch_to.window(browser.window_handles[1])
# #     browser.get("https://www.diretta.it/partita/{}/#/informazioni-partita/informazioni-partita".format(match_id))
# #     # Scrape
# #     # Close the new tab
# #     browser.close()
# #     # Switch to old tab
# #     browser.switch_to.window(browser.window_handles[0])
# #     return ''
#
#
# # def scrape_data_from_season_page(browser: webdriver, bs: BeautifulSoup, season: str) -> pd.DataFrame:
# #     print('Scraping data...')
# #     # Find first match data
# #     match_parent_tag = bs.find('div', {'id': lambda x: x and x.startswith('g_1_')})
# #     match_data_as_list = tokenize_match_data(match_parent_tag)
# #     match_id = get_match_id_from_parent_tag(match_parent_tag)
# #     match_data_as_list.append(match_id)
# #     match_data_as_list.append(get_match_teams_data(browser, bs, match_id))
# #     # Handle special cases
# #     match_data_as_list = handle_special_cases(match_data_as_list)
# #     # Format date
# #     match_data_as_list[0] = format_date(match_data_as_list[0], season)
# #     # Construct DataFrame
# #     match_data = pd.DataFrame([match_data_as_list], columns=cols)
# #     season_data = match_data
# #     # Find all next match data
# #     while True:
# #         try:
# #             match_parent_tag = match_parent_tag.find_next('div', {'id': lambda x: x and x.startswith('g_1_')})
# #             match_data_as_list = tokenize_match_data(match_parent_tag)
# #             # Handle special cases
# #             if is_match_won_by_forfeit(match_data_as_list):
# #                 continue
# #             match_data_as_list = handle_special_cases(match_data_as_list)
# #             # Append match id
# #             match_data_as_list.append(get_match_id_from_parent_tag(match_parent_tag))
# #             # Format date
# #             match_data_as_list[0] = format_date(match_data_as_list[0], season)
# #             # Append DataFrame
# #             match_data = pd.DataFrame([match_data_as_list], columns=cols)
# #             season_data = pd.concat([season_data, match_data], ignore_index=True)
# #         except AttributeError:
# #             break
# #     print('Done')
# #     return season_data
#
#
# def init_headless_browser():
#     chrome_options.add_argument("--headless")
#     # chrome_options.add_argument("--start-maximised")
#     return webdriver.Chrome(
#         executable_path="D:\\Windows\\Programmi\\phantomjs-2.1.1\\chromedriver_win32\\chromedriver.exe",
#         options=chrome_options)
#
#
# def find_match_parent_tag(bs: BeautifulSoup):
#     return bs.find('div', {'id': lambda x: x and x.startswith('g_1_')})
#
#
# def find_next_parent_tag(current_match_parent_tag):
#     return current_match_parent_tag.find_next('div', {'id': lambda x: x and x.startswith('g_1_')})
#
#
# def scrape_matches_ids_from_season_page(bs: BeautifulSoup) -> list[str]:
#     print('Scraping matches ids...')
#     matches_ids = []
#     # Find first match id
#     match_parent_tag = find_match_parent_tag(bs)
#     match_id = get_match_id_from_parent_tag(match_parent_tag)
#     matches_ids.append(match_id)
#     # Find all subsequent matches ids
#     while True:
#         try:
#             match_parent_tag = find_next_parent_tag(match_parent_tag)
#             match_id = get_match_id_from_parent_tag(match_parent_tag)
#             matches_ids.append(match_id)
#         except AttributeError:
#             break
#     print('Done')
#     return matches_ids
#
#
# # def open_url_in_new_tab_and_parse(browser: webdriver, url: str) -> BeautifulSoup:
# #     # Open a new tab
# #     browser.execute_script("window.open('');")
# #     # Switch to the new tab and open new URL
# #     browser.switch_to.window(browser.window_handles[1])
# #     return get_and_parse_url(url)
#
#
# def close_current_tab(browser: webdriver):
#     browser.close()
#     # Switch to old tab
#     browser.switch_to.window(browser.window_handles[0])
#
#
# def get_and_parse_url(browser: webdriver, url: str) -> BeautifulSoup:
#     browser.get(url)
#     html = browser.page_source
#     return BeautifulSoup(html, 'html.parser')
#
#
# def get_wait_and_parse_url(browser: webdriver, url: str, class_name: str = '') -> BeautifulSoup:
#     browser.get(url)
#     if class_name != '':
#         WebDriverWait(browser, 5).until(
#             expected_conditions.visibility_of_element_located((By.CLASS_NAME, class_name)))
#     html = browser.page_source
#     return BeautifulSoup(html, 'html.parser')
#
#
# def find_next_player_tag(bs: BeautifulSoup, player_tag):
#     return player_tag.find_next('a',
#                                 {'href': lambda x: x and x.startswith('/giocatore/'), 'class': 'lf__participantName'})
#
#
# def scrape_match_players(browser: webdriver, match_id: str):
#     match_players_info_url = "https://www.diretta.it/partita/{}/#/informazioni-partita/formazioni".format(match_id)
#     bs = get_wait_and_parse_url(browser, match_players_info_url, 'lf__lineUp')
#     # browser.get("https://www.diretta.it/partita/{}/#/informazioni-partita/formazioni".format(match_id))
#     # WebDriverWait(browser, 10).until(
#     #     expected_conditions.visibility_of_all_elements_located((By.CLASS_NAME, 'lf__lineUp')))
#     # html = browser.page_source
#     # bs = BeautifulSoup(html, 'html.parser')
#     players = []
#     player_tag = bs.find('a', {'href': lambda x: x and x.startswith('/giocatore/'), 'class': 'lf__participantName'})
#     player = player_tag.get_attribute_list('href')[0].split('/')[2].replace('-', ' ')
#     players.append(player)
#     while True:
#         try:
#             player_tag = find_next_player_tag(bs, player_tag)
#             player = player_tag.get_attribute_list('href')[0].split('/')[2].replace('-', ' ')
#             players.append(player)
#         except AttributeError:
#             break
#     return players
#
#
# def scrape_match_basic_data(browser: webdriver, match_id: str):
#     match_info_url = "https://www.diretta.it/partita/{}/#/informazioni-partita/informazioni-partita".format(match_id)
#     bs = get_wait_and_parse_url(browser, match_info_url, 'smv__verticalSections.section')
#     match_round = int(bs.find('a', {'href': '/serie-a/'}).getText().replace('Serie A - Giornata ', ''))
#     match_datetime = bs.find('div', {'class': 'duelParticipant__startTime'}).getText()
#     match_team_tag = bs.find('div', {'class': 'participant__participantName participant__overflow'})
#     match_home_team = match_team_tag.getText()
#     match_away_team = match_team_tag.find_next('div', {
#         'class': 'participant__participantName participant__overflow'}).getText()
#     score = bs.find('div', {'class': 'detailScore__wrapper'}).getText()
#     match_home_team_score = score.split('-')[0]
#     match_away_team_score = score.split('-')[1]
#     fh_score = bs.find('div', {'class': 'smv__incidentsHeader section__title'}).getText().replace('1 Tempo', '')
#     match_fh_home_team_score = fh_score.split('-')[0].replace(' ', '')
#     match_fh_away_team_score = fh_score.split('-')[1].replace(' ', '')
#     return [
#         match_round,
#         match_datetime,
#         match_home_team,
#         match_away_team,
#         match_home_team_score,
#         match_away_team_score,
#         match_fh_home_team_score,
#         match_fh_away_team_score,
#     ]
#
#
# def scrape_matches_data(browser: webdriver, matches_ids: list[str]) -> pd.DataFrame:
#     season_data = pd.DataFrame(columns=cols1)
#     for match_id in matches_ids:
#         match_basic_data = scrape_match_basic_data(browser, match_id)
#         match_players = scrape_match_players(browser, match_id)
#         match_data_as_list = match_basic_data + match_players
#         match_data = pd.DataFrame([match_data_as_list], columns=cols1)
#         pd.concat([season_data, match_data], ignore_index=True)
#     return season_data
#
#
# def scrape_data() -> pd.DataFrame:
#     browser = init_headless_browser()
#     years = np.arange(2005, 2006, 1)
#     seasons = np.array(["{}-{}".format(years[i], years[i] + 1) for i in range(years.size)])
#     seasons_data = pd.DataFrame(columns=cols)
#     for season in seasons:
#         bs = get_and_parse_season_page(browser, season)
#         matches_ids = scrape_matches_ids_from_season_page(bs)
#         season_data = scrape_matches_data(browser, matches_ids)
#         seasons_data = pd.concat([seasons_data, season_data])
#     browser.quit()
#     return seasons_data
