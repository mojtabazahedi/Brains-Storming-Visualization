import mysql.connector
from collections import Counter
from gensim.parsing.preprocessing import remove_stopwords
import json

mydb = mysql.connector.connect(
    host="localhost",
    user='root',
    password='',
    database='myproject'
)


myCursor = mydb.cursor()
myCursor.execute("SELECT input FROM inputapp_input")

# convert tuple to string
dbString =''
myResult = myCursor.fetchall()
for x in myResult:
    toStr = ''.join(x)
    dbString += toStr

# remove stop words from text and also covert to lower case
filtered_sentence = remove_stopwords(dbString.lower())
split_it = filtered_sentence.split()
Counter = Counter(split_it)
most_occur = Counter.most_common()
print(most_occur)
with open("most.json", "w") as mst:
    json.dump(most_occur, mst)






