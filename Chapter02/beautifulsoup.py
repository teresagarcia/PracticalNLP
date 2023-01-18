from pprint import pprint
from bs4 import BeautifulSoup
from urllib.request import urlopen

myurl = "https://yasashiiuta05.blogspot.com/2022/08/letra-nabeel-shaukat-aima-baig-ja-tujhe.html" # specify the url
html = urlopen(myurl).read() # query the website so that it returns a html page
soupified = BeautifulSoup(html, 'html.parser') # parse the html in the 'html' variable, and store it in Beautiful Soup format

pprint(soupified.prettify()[:2000]) # to get an idea of the html structure of the webpage

# print(soupified.title) # to get the title of the web page

post = soupified.find("div", {"class": "post"})

title = post.find("h3", {"class": "post-title entry-title"}) # find the nevessary tag and class which it belongs to
title_text = title.get_text().strip()
print("Título: \n", title_text)

post_body = post.find("div", {"class": "post-body entry-content float-container"})

video = post_body.find("iframe")['src']
print("Link vídeo: \n", video)

post_text = post_body.select('div')[2].get_text().strip()
print("Texto de la entrada: \n", post_text)