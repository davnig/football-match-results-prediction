import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

chrome_options = Options()
cols = ['date', 'time', 'team_A', 'team_B', 'goal_A', 'goal_B', 'fh_goal_A', 'fh_goal_B', 'match_id']


def get_and_parse_season_page(browser: webdriver, season):
    print('Parsing page for season {}...'.format(season))
    url = "https://www.diretta.it/serie-a-{}/risultati".format(season)
    browser.get(url)
    # OPTIONAL: CLOSE BANNER
    # try:
    #     browser.find_element_by_id('onetrust-reject-all-handler').click()
    # except ElementClickInterceptedException:
    #     pass
    for i in range(3):
        show_more_btn = browser.find_element_by_css_selector('a.event__more')
        browser.execute_script("arguments[0].scrollIntoView(true);", show_more_btn)
        WebDriverWait(browser, 10).until(
            expected_conditions.element_to_be_clickable((By.CSS_SELECTOR, 'a.event__more')))
        show_more_btn.click()
    html = browser.page_source
    return BeautifulSoup(html, 'html.parser')


def tokenize_match_data(match_data_parent_tag) -> list[str]:
    match_data = ''
    for child in match_data_parent_tag.findChildren():
        match_data += child.getText() + ' '
    return match_data.split()


def handle_special_cases(match_data_list):
    # Change 'ACR Messina' to 'Messina'
    if 'ACR' in match_data_list:
        match_data_list.remove('ACR')
    return match_data_list


def format_date(date, season):
    month = int(date.split('.')[1])
    if 8 <= month <= 12:
        date = date + season.split('-')[0]
    else:
        date = date + season.split('-')[1]
    return date


def is_match_won_by_forfeit(match_data_list):
    return 'A' in match_data_list and 'Tav.' in match_data_list


def get_match_id_from_parent_tag(match_parent_tag):
    return match_parent_tag.get_attribute_list('id')[0].replace('g_1_', '')


def get_match_teams_data(browser, bs, match_id) -> str:
    # Open a new tab
    browser.execute_script("window.open('');")
    # Switch to the new tab and open new URL
    browser.switch_to.window(browser.window_handles[1])
    browser.get("https://www.diretta.it/partita/{}/#/informazioni-partita/informazioni-partita".format(match_id))
    # Scrape
    # Close the new tab
    browser.close()
    # Switch to old tab
    browser.switch_to.window(browser.window_handles[0])
    return ''


def scrape_data_from_season_page(browser: webdriver, bs: BeautifulSoup, season: str) -> pd.DataFrame:
    print('Scraping data...')
    # Find first match data
    match_parent_tag = bs.find('div', {'id': lambda x: x and x.startswith('g_1_')})
    match_data_as_list = tokenize_match_data(match_parent_tag)
    match_id = get_match_id_from_parent_tag(match_parent_tag)
    match_data_as_list.append(match_id)
    match_data_as_list.append(get_match_teams_data(browser, bs, match_id))
    # Handle special cases
    match_data_as_list = handle_special_cases(match_data_as_list)
    # Format date
    match_data_as_list[0] = format_date(match_data_as_list[0], season)
    # Construct DataFrame
    match_data = pd.DataFrame([match_data_as_list], columns=cols)
    season_data = match_data
    # Find all next match data
    while True:
        try:
            match_parent_tag = match_parent_tag.find_next('div', {'id': lambda x: x and x.startswith('g_1_')})
            match_data_as_list = tokenize_match_data(match_parent_tag)
            # Handle special cases
            if is_match_won_by_forfeit(match_data_as_list):
                continue
            match_data_as_list = handle_special_cases(match_data_as_list)
            # Append match id
            match_data_as_list.append(get_match_id_from_parent_tag(match_parent_tag))
            # Format date
            match_data_as_list[0] = format_date(match_data_as_list[0], season)
            # Append DataFrame
            match_data = pd.DataFrame([match_data_as_list], columns=cols)
            season_data = pd.concat([season_data, match_data], ignore_index=True)
        except AttributeError:
            break
    print('Done')
    return season_data


def init_headless_browser():
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximised")
    return webdriver.Chrome(
        executable_path="D:\\Windows\\Programmi\\phantomjs-2.1.1\\chromedriver_win32\\chromedriver.exe",
        options=chrome_options)


def scrape_data() -> pd.DataFrame:
    browser = init_headless_browser()
    years = np.arange(2005, 2021, 1)
    seasons = np.array(["{}-{}".format(years[i], years[i] + 1) for i in range(years.size)])
    seasons_data = pd.DataFrame(columns=cols)
    for season in seasons:
        bs = get_and_parse_season_page(browser, season)
        season_data = scrape_data_from_season_page(browser, bs, season)
        seasons_data = pd.concat([seasons_data, season_data])
    browser.quit()
    return seasons_data
