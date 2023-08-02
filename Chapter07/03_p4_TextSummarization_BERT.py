

import os
from summarizer import Summarizer
text = open(f"{os.getcwd()}/data/Chapter05/nlphistory.txt").read()

model = Summarizer()

print("Without Coreference:")
result = model(text, min_length=200,ratio=0.01)
full = ''.join(result)
print(full)


# print("With Coreference:")
# # handler = CoreferenceHandler(greedyness=.35)

# model = Summarizer(sentence_handler=handler)
# result = model(text, min_length=200,ratio=0.01)
# full = ''.join(result)
# print(full)