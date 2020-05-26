from bs4 import BeautifulSoup
import requests


base_url = "https://namemc.com"
top_skins = "/minecraft-skins/trending/top"
rand_skins = "/minecraft-skins/random"
texture_link = "/texture/"


def download_skin(skin_name):
    print("Downloading: " + skin_name)
    img_data = requests.get(base_url + texture_link + skin_name + ".png").content
    with open('output/' + skin_name + '.png', 'wb') as handler:
        handler.write(img_data)


def get_top():
    cur_page = 1
    while cur_page <= 100:
        print("===== Page " + str(cur_page) + " =====")
        page = requests.get(base_url + top_skins + "?page=" + str(cur_page))
        soup = BeautifulSoup(page.text, "html.parser")
        cards = soup.find_all('div', class_='card')
        for card in cards:
            skin = card.find('a')
            name = skin.get('href')
            download_skin(name.split('/')[-1])

        cur_page += 1
    print("Finished!")


def get_rand(num):
    cur_page = 1
    while cur_page <= num:
        print("===== Rand Page " + str(cur_page) + " =====")
        page = requests.get(base_url + rand_skins)
        soup = BeautifulSoup(page.text, "html.parser")
        cards = soup.find_all('div', class_='card')
        for card in cards:
            skin = card.find('a')
            name = skin.get('href')
            download_skin(name.split('/')[-1])

        cur_page += 1
    print("Finished!")


# get_top()
get_rand(5)
