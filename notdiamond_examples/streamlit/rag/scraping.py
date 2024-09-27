import logging
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests

_LOGGER = logging.getLogger(__name__)

def scrape_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Explicitly set encoding in case it's not detected properly
        response.encoding = response.apparent_encoding

        # Parse the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Collect content from <p>, <h1>, <h2>, <h3> tags for a more comprehensive scrape
        # <div> tags are excluded to avoid redundant content
        content = []
        for tag in ['p', 'h1', 'h2', 'h3']:
            for element in soup.find_all(tag):
                text = element.get_text(separator="\n", strip=True)
                if text:  # Only add non-empty text
                    content.append(text)

        # Join the extracted content into a single string
        main_content = "\n".join(content)

        # Extract all the links from the page
        links = {a.get('href') for a in soup.find_all('a', href=True)}
        links = {urljoin(url, link) for link in links if link and link not in ('#', '/') and link[0] != "#"}

        _LOGGER.info(f"Scraping URL: {url}\n")
        _LOGGER.info("Main Content (Preview):\n", main_content[:1000])
        _LOGGER.info("\n" + "="*50 + "\n")

        return main_content, links, soup
    except requests.exceptions.RequestException as e:
        _LOGGER.error(f"Error fetching {url}: {e}")
        return None, [], None


# Function to scrape the initial page and follow all links containing 'example.com'
def scrape_website(
    url: str,
    domain_filter: Any,
    follow_links: bool,
    exclude_header_footer: bool = True,
    max_depth: int = 1,
) -> List[Dict[str, str]]:
    visited = set()
    to_visit = [(url, 0)]
    scraped_data = []

    while to_visit:
        current_url, current_depth = to_visit.pop(0)
        print(current_url)
        if current_url not in visited and current_depth <= max_depth:
            visited.add(current_url)
            content, links, soup = scrape_content(current_url)

            if content:
                scraped_data.append({"URL": current_url, "Content": content})
                _LOGGER.info(f"Scraped content from: {current_url}")
            else:
                _LOGGER.warning(f"No content found at: {current_url}")

            if follow_links and current_depth < max_depth:
                if exclude_header_footer and soup:
                    header_links = [a.get('href') for a in soup.select('header a, [class*="header"] a, [id*="header"] a')]
                    footer_links = [a.get('href') for a in soup.select('footer a, [class*="footer"] a, [id*="footer"] a')]
                    links = [link for link in links if link not in header_links and link not in footer_links]
                    links = links[:30]

                filtered_links = {link for link in links if domain_filter in urlparse(link).netloc}
                print(filtered_links)
                to_visit.extend([(link, current_depth + 1) for link in filtered_links if link not in visited])
                _LOGGER.info(f"Found {len(filtered_links)} links to follow from: {current_url}")

    return scraped_data

