from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests
import io
from PIL import Image
import time
import os

DOWNLOAD_PATH = r"C:\Users\hp\OneDrive\Documents\AI\laptopMonitorReco\2"

options = webdriver.ChromeOptions()
service = Service(ChromeDriverManager().install())
wd = webdriver.Chrome(service=service, options=options)

def get_images_from_google(wd, delay, max_images):
    def scroll_down(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)
    
    image_urls = set()
    for i in range(3, 10):
        url = f"https://www.ebay.com/b/Computer-Monitors/80053/bn_317528?_pgn={i}"
        print(f"Fetching URL: {url}")
       
        wd.get(url)
        scroll_down(wd)

        try:
            # Wait for the thumbnails to be present
            WebDriverWait(wd, delay).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "s-item__image-wrapper"))
            )
            thumbnails = wd.find_elements(By.CLASS_NAME, "s-item__image-wrapper")
            print(f"Found {len(thumbnails)} thumbnails on page {i}")

            for img in thumbnails[:max_images]:
                try:
                    img.click()
                    time.sleep(delay)
                except Exception as e:
                    print(f"Error clicking thumbnail: {e}")
                    continue

                try:
                    WebDriverWait(wd, delay).until(
                        EC.presence_of_all_elements_located((By.TAG_NAME, "img"))
                    )
                    images = wd.find_elements(By.TAG_NAME, "img")
                    for image in images:
                        src = image.get_attribute('src')
                        if src and 'http' in src:
                            image_urls.add(src)
                            print(f"Found {len(image_urls)} images")
                            if len(image_urls) >= max_images:
                                break
                except Exception as e:
                    print(f"Error finding images: {e}")

           

        except Exception as e:
            print(f"Error fetching thumbnails: {e}")
        
    return image_urls

def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = os.path.join(download_path, file_name)

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            return

        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        print(f"Success: {file_path}")
    except Exception as e:
        print('FAILED -', e)

# Ensure the download directory exists
if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)

urls = list(get_images_from_google(wd, 5, 10))  

for i, url in enumerate(urls, 1):
    download_image(DOWNLOAD_PATH, url, f"{i}.jpg")

wd.quit()
import scrapy

class AcncSpider(scrapy.Spider):
    name = 'acnc_spider'
    start_urls = ['https://www.acnc.gov.au/charity/charities?items_per_page=100&page=2&f[]=countries%3A3411872625&f[]=size%3ALarge']

    def parse(self, response):
        for card in response.css('div.search-result__item'):
            charity_name = card.css('h3::text').get().strip()
            charity_link = response.urljoin(card.css('a::attr(href)').get())
            
            yield scrapy.Request(
                charity_link,
                callback=self.parse_detail_page,
                meta={'charity_name': charity_name}
            )

    def parse_detail_page(self, response):
        email_tag = response.css('a[href^="mailto:"]::text').get()
        email = email_tag.strip() if email_tag else 'Not Available'
        yield {
            'Charity Name': response.meta['charity_name'],
            'Email': email,
            'Detail Page': response.url
        }
