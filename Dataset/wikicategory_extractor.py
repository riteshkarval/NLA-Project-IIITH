#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wikipediaapi


# In[2]:


wiki_wiki = wikipediaapi.Wikipedia(language='en',extract_format=wikipediaapi.ExtractFormat.WIKI)


# In[3]:


def print_categorymembers(categorymembers,titles, level=0, max_level=1):
    for c in categorymembers.values():
        titles.append(c.title)
        if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
             print_categorymembers(c.categorymembers,titles, level=level + 1, max_level=max_level)
    return titles


# In[33]:


phy_cat = 'Category:Physics'
chem_cat = 'Category:Chemistry'
math_cat = 'Category:Mathematics'
bio_cat = 'Category:Biology'

cat = wiki_wiki.page(phy_cat)
titles = print_categorymembers(cat.categorymembers, titles)
error_titles = []
for each_title in titles:
    try:
        page_text = wiki_wiki.page(each_title).text
        page_path = 'physicsdocs/'+ each_title
        with open(page_path, 'a') as the_file:
            the_file.write(page_text)
    except:
        print(each_title)
        error_titles.append(each_title)
        
cat = wiki_wiki.page(chem_cat)
titles = print_categorymembers(cat.categorymembers, titles)
error_titles = []
for each_title in titles:
    try:
        page_text = wiki_wiki.page(each_title).text
        page_path = 'chemestrydocs/'+ each_title
        with open(page_path, 'a') as the_file:
            the_file.write(page_text)
    except:
        print(each_title)
        error_titles.append(each_title)
        
cat = wiki_wiki.page(math_cat)
titles = print_categorymembers(cat.categorymembers, titles)
error_titles = []
for each_title in titles:
    try:
        page_text = wiki_wiki.page(each_title).text
        page_path = 'mathdocs/'+ each_title
        with open(page_path, 'a') as the_file:
            the_file.write(page_text)
    except:
        print(each_title)
        error_titles.append(each_title)
        
cat = wiki_wiki.page(bio_cat)
titles = print_categorymembers(cat.categorymembers, titles)
error_titles = []
for each_title in titles:
    try:
        page_text = wiki_wiki.page(each_title).text
        page_path = 'biologydocs/'+ each_title
        with open(page_path, 'a') as the_file:
            the_file.write(page_text)
    except:
        print(each_title)
        error_titles.append(each_title)

