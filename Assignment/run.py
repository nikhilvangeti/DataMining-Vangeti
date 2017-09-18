import urllib.request
from bs4 import BeautifulSoup
import os

#requesting the url and getting it readt to be scraped
thePage=urllib.request.urlopen("http://www.imdb.com/chart/top")
soupData=BeautifulSoup(thePage,"html.parser")

#Listing values to be entered into .csv file
moviesList=""

#scraping values from a table, starting with a row and scraping data values from each row
for table in soupData.findAll('tr'):
    #scraping data values from class titleColumn, one of many classes in the row
    for data in table.findAll('td',{"class":"titleColumn"}):
        #moviesList=moviesList+"\n"+data.find('a').text
        #scraping title and the year of release with respect to their tags
        print(data.find('a').text)
        print(data.find('span').text)
        # adding all values into the list
        moviesList=moviesList+"\n"+data.find('a').text+"    "+data.find('span').text
        
#creating a .csv file and getting it ready to be written        
file=open(os.path.expanduser("MoviesList.csv"),"wb")
header="Movies, Year Of Reslease"
#header for headings and next printing the list into a .csv file
file.write(bytes(header,encoding="ascii",errors="ignore"))
file.write(bytes(moviesList,encoding="ascii",errors="ignore"))
#Finally printing the scraped values onto the console
print(moviesList)
