import os
from summa import summarizer
from summa import keywords

text = open(f"{os.getcwd()}/data/Chapter05/nlphistory.txt").read()

print("Summary:")
print (summarizer.summarize(text,ratio=0.3))

