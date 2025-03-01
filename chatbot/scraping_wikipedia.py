import glob
import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

DATA_DIR = "data"
FIRST_URL = "https://id.wikipedia.org/wiki/Minecraft"
KEYWORD = "minecraft"
THRESHOLD = 10
MAX_DEPTH = 1

BLACKLIST = [
    "Berkas:",
    "Wikipedia:",
    "Templat:",
    "Istimewa:",
    "Kategori:",
    "Pembicaraan:",
]


def valid_link(text: str):
    if not isinstance(text, str):
        return False
    if not text.startswith("/wiki/"):
        return False
    for w in BLACKLIST:
        if w in text:
            return False
    return True


pending_urls = []
visited_urls = set()
df_pages = pd.DataFrame(columns=["Title", "URL", "Content"])

pending_urls.append((FIRST_URL, 0))
while len(pending_urls) > 0:
    url, depth = pending_urls.pop()

    if url in visited_urls:
        continue
    visited_urls.add(url)

    try:
        response = requests.get(url=url)
        if response.status_code != 200:
            print(f"Error fetching {url}: Status code {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")

        title_text = soup.find(id="firstHeading").get_text(" ", strip=True)
        content = soup.find(id="mw-content-text")

        content_text = content.get_text(" ", strip=True).lower()
        count = content_text.count(KEYWORD)

        has_keyword = count >= THRESHOLD
        if has_keyword:
            df_pages.loc[len(df_pages)] = {
                "Title": title_text,
                "URL": url,
                "Content": str(soup),
            }
            print(
                f"No: {len(df_pages)}, Title: {title_text}, URL: {url}, Count: {count}"
            )

        if (not has_keyword) or (depth + 1 > MAX_DEPTH):
            continue

        # cari semua link untuk lanjut scraping
        links = soup.find(id="bodyContent").find_all("a")
        links = map(lambda x: x.get("href"), links)
        links = filter(valid_link, links)
        links = map(lambda x: x.split("#")[0], links)  # hapus hash
        links = list(set(links))  # hapus duplikat
        for link in links:
            full_url = "https://id.wikipedia.org" + link
            pending_urls.append((full_url, depth + 1))

    except Exception as e:
        print("Error scraping", url, ":", e)


def clean_html_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # hapus elemen berdasarkan id, class, atau tag yang diketahui tidak diperlukan
    elements_to_remove = [
        "nav",
        "aside",
        "#vector-main-menu",
        "#p-navigation",
        "#p-Komunitas",
        "#p-Wikipedia",
        "#p-Bagikan",
        "#p-tb",
        "#p-coll-print_export",
        "#p-wikibase-otherprojects",
        ".noprint",
        ".mw-indicators",
        ".hatnote",
        ".vector-menu",
        ".vector-dropdown",
        ".vector-header-container",
        ".vector-page-toolbar",
        'a.mw-jump-link[href="#bodyContent"]',
        ".printfooter",
        ".mw-footer-container",
        ".vector-settings",
        ".catlinks",
        ".mw-hidden-catlinks.mw-hidden-cats-hidden",
        '[src$=".png"]',
        '[src$=".jpg"]',
        '[src$=".jpeg"]',
        '[src$=".gif"]',
        '[src$=".svg"]',
        ".mw-editsection",
        "figure",
        "title",
        ".ambox",
    ]
    for selector in elements_to_remove:
        for element in soup.select(selector):
            element.decompose()

    # hapus semua elemen <link>
    for link in soup.find_all("link"):
        link.decompose()
    for a in soup.find_all("a", href=True):
        a.replace_with(a.get_text())

    # hapus italic sama bold
    for i in soup.find_all("i"):
        i.unwrap()
    for b in soup.find_all("b"):
        b.unwrap()

    # hapus semua elemen <meta>
    for meta in soup.find_all("meta", content=True):
        meta.decompose()

    # hapus semua elemen <script>
    for script in soup.find_all("script"):
        script.decompose()

    # hapus referensi dalam bentuk [1]
    for sup in soup.find_all("sup", {"class": "reference"}):
        sup.decompose()

    # hapus bagian "Referensi" dll
    lihat_pula_heading = soup.find("h2", {"id": "Lihat_pula"})
    if lihat_pula_heading:
        current_element = lihat_pula_heading.parent
        while current_element:
            next_element = current_element.find_next_sibling()
            current_element.decompose()
            current_element = next_element
    referensi_heading = soup.find("h2", {"id": "Referensi"})
    if referensi_heading:
        current_element = referensi_heading.parent
        while current_element:
            next_element = current_element.find_next_sibling()
            current_element.decompose()
            current_element = next_element

    for tag in soup.find_all(True):
        for attr in ["class", "style"]:
            if attr in tag.attrs:
                del tag[attr]

    return str(soup)


if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for index, row in df_pages.iterrows():
    url = row["URL"]
    title: str = row["Title"]
    content = row["Content"]

    cleaned_html_content = clean_html_content(content)
    # print(cleaned_html_content)

    md_content = md(cleaned_html_content, heading_style="ATX", bullet_style="*")
    # print(md_content)

    md_content = re.sub(
        r"!\[\]\(.*login.wikimedia.org.*\)",
        "",
        md_content,
    )
    md_clean = re.sub(r"\n{3,}", "\n\n", md_content).strip()

    sanitized_title = (
        title.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("*", "-")
        .replace("?", "-")
        .replace('"', "-")
        .replace("<", "-")
        .replace(">", "-")
        .replace("|", "-")
    )

    filename = os.path.join(DATA_DIR, f"{sanitized_title}.md")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(md_clean)

    print(f"{index+1}. Konten dari {url} telah dibersihkan dan diproses.")
