from bs4 import BeautifulSoup as Soup
import requests
import pdb

# url = "https://www.northwestern.edu/international/international-students/"

def fetch_text_from_url(url):
    response = requests.get(url)

    soup = Soup(response.text, 'html.parser')

    content_divs = soup.find_all('div', class_='content')

    def remove_text_by_id(tag, target_id):
        target_tag = tag.find(id=target_id)
        if target_tag:
            target_tag.clear()


    for div in content_divs:
        remove_text_by_id(div, "breadcrumbs")

    extracted_content = []
    for div in content_divs:
        # content_text = div.get_text(strip=True)
        # tags_to_extract = [tag for tag in div.find_all(True)]
        
        # extracted_content.append('\n'.join(tag.get_text() for tag in tags_to_extract))
        extracted_content.append(div.get_text(strip=True))
        # print(extracted_content)
    return extracted_content

# pdb.set_trace()
# raw_text = fetch_text_from_url("https://www.northwestern.edu/international/international-students/")
# text_chunks = get_text_chunks(raw_text[0])

# print(len(text_chunks))