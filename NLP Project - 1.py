#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch')
get_ipython().system('pip install transformers')
get_ipython().system('pip install bs4')
get_ipython().system('pip install sentencepiece')


# In[3]:


from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
import torch


# In[4]:


model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


# In[5]:


url = "https://uk.finance.yahoo.com/news/avalo-therapeutics-nasdaq-avtx-investors-104344116.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuYmluZy5jb20v&guce_referrer_sig=AQAAAHT8xpQi3wTiC-LAfIs54o76H-X2Rqx-9SsAEC2E-O__chNdZLKVpPBO_mrpWoH4g43io8wPJkag_8GLIoGoo6wgFkuqppavxAsGJmV8vPD62LIMCYLLxNlP6RgwBXap39YM30AeOe9XPkV-1qvrj1WZkHm_x6d8NB_aKQnLqkRu"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
paragraphs = soup.find_all('p')


# In[7]:


paragraphs


# In[14]:


text = [paragraph.text for paragraph in paragraphs]
words = ' '.join(text).split(' ')[:400]
ARTICLE = ' '.join(words)


# In[20]:


ARTICLE


# In[25]:


input_ids = tokenizer.encode(ARTICLE, return_tensors = 'pt')
output = model.generate(input_ids,max_length = 100, num_beams=10,early_stopping=False)
summary =tokenizer.decode(output[0],skip_special_tokens=True)


# In[26]:


summary


# In[33]:


monitored_tickers = ['CERC','INFI','BTC']


# In[79]:


def search_for_stock_news_url(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs


# In[103]:


raw_urls = {ticker:search_for_stock_news_url(ticker) for ticker in monitored_tickers}


# In[81]:


import re


# In[97]:


exclude_list = ['maps','policies','preferences','accounts','support']


# In[100]:


def strip_unwanted_urls(urls,exclude_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
            print(res)
        return list(val)


# In[101]:


cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}
cleaned_urls


# In[102]:


raw_urls


# In[ ]:




