import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

chrome_options = Options()


def get_and_parse_season_page(browser, season):
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


def scrape_data_by_season_page(bs):
    print('Scraping data...')
    season_data = []
    # FIND FIRST DIV CONTAINING MATCH DATA
    match_data_parent = bs.find('div', {'id': lambda x: x and x.startswith('g_1_')})
    match_data = ''
    for child in match_data_parent.findChildren():
        match_data += child.getText() + ' '
    season_data.append(match_data.split())
    # ITERATE OVER THE NEXT SAME ELEMENTS
    while True:
        try:
            match_data_parent = match_data_parent.find_next('div', {'id': lambda x: x and x.startswith('g_1_')})
            match_data = ''
            for child in match_data_parent.findChildren():
                match_data += child.getText() + ' '
            season_data.append(match_data.split())
        except AttributeError:
            break
    print(season_data)
    return season_data


def init_headless_browser():
    chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--start-maximised")
    return webdriver.Chrome(
        executable_path="D:\\Windows\\Programmi\\phantomjs-2.1.1\\chromedriver_win32\\chromedriver.exe",
        options=chrome_options)


def scrape_data():
    browser = init_headless_browser()
    years = np.arange(2005, 2021, 1)
    seasons = np.array(["{}-{}".format(years[i], years[i] + 1) for i in range(years.size)])
    seasons_data = []
    # for i in range(seasons.size):
    for i in range(2):
        bs = get_and_parse_season_page(browser, seasons[i])
        seasons_data = [scrape_data_by_season_page(bs)]
    browser.quit()
    return seasons_data
