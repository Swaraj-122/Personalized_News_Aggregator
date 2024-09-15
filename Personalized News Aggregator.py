import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from fastapi import FastAPI
import pandas as pd
import spacy
from collections import Counter


# Function to scrape a news website (e.g., BBC)
def scrape_bbc_news():
    url = 'https://www.bbc.com/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    articles = []
    for item in soup.find_all('div', class_='gs-c-promo'):
        title = item.find('h3').get_text() if item.find('h3') else None
        summary = item.find('p').get_text() if item.find('p') else None
        date = "Today"  # You might need more specific date scraping
        source = 'BBC'
        link = item.find('a')['href']
        if link and title:
            articles.append([title, summary, date, source, link])

    return articles

# Save the scraped data into a CSV
def save_to_csv(articles, filename='news_articles.csv'):
    df = pd.DataFrame(articles, columns=['Title', 'Summary', 'Publication Date', 'Source', 'URL'])
    df.to_csv(filename, index=False)


def scrape_cnn_news():
    driver = webdriver.Chrome()  # Make sure to have ChromeDriver installed
    driver.get('https://www.cnn.com/world')

    articles = []
    titles = driver.find_elements_by_tag_name('h3')
    for title in titles[:10]:  # Limiting to first 10 articles
        summary = title.text
        url = title.find_element_by_tag_name('a').get_attribute('href')
        articles.append([summary, "N/A", "Today", "CNN", url])

    driver.quit()
    return articles

all_articles = bbc_articles + scrape_cnn_news()
save_to_csv(all_articles)



# Load pre-trained spaCy model
nlp = spacy.load('en_core_web_sm')

# Simple categorization by keyword matching (or you can use advanced topic modeling)
categories = {
    'Politics': ['government', 'election', 'politics'],
    'Technology': ['technology', 'tech', 'innovation'],
    'Sports': ['sports', 'game', 'team']
}

# Function to categorize articles
def categorize_articles(articles):
    for article in articles:
        doc = nlp(article[1])  # Using article summary for categorization
        tokens = [token.text.lower() for token in doc]
        category_count = {cat: sum([1 for word in keywords if word in tokens]) for cat, keywords in categories.items()}
        category = max(category_count, key=category_count.get) if category_count else 'Other'
        article.append(category)
    return articles

# Update CSV with categorized articles
def update_csv_with_categories(articles, filename='news_articles.csv'):
    df = pd.read_csv(filename)
    df['Category'] = [article[-1] for article in articles]
    df.to_csv(filename, index=False)

categorized_articles = categorize_articles(all_articles)
update_csv_with_categories(categorized_articles)


app = FastAPI()

# Load data
df = pd.read_csv('news_articles.csv')

# GET all articles with optional filtering by date range and category
@app.get("/articles")
def get_articles(category: str = None, start_date: str = None, end_date: str = None):
    filtered_df = df.copy()
    if category:
        filtered_df = filtered_df[filtered_df['Category'].str.contains(category, case=False, na=False)]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Publication Date'] >= start_date) & (filtered_df['Publication Date'] <= end_date)]
    return filtered_df.to_dict(orient='records')

# GET specific article by ID
@app.get("/articles/{article_id}")
def get_article(article_id: int):
    article = df.iloc[article_id]
    return article.to_dict()

# GET search articles by keyword
@app.get("/search")
def search_articles(keyword: str):
    filtered_df = df[df['Title'].str.contains(keyword, case=False, na=False)]
    return filtered_df.to_dict(orient='records')

# Run the FastAPI server using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
