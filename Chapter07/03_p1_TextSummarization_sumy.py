import nltk

#Code to summarize a given webpage using Sumy's TextRank implementation. 
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer

num_sentences_in_summary = 7
url = "https://en.wikipedia.org/wiki/Music_of_Punjab"
parser = HtmlParser.from_url(url, Tokenizer("english"))

summarizer_list=("TextRankSummarizer:","LexRankSummarizer:","LuhnSummarizer:","LsaSummarizer") #list of summarizers
summarizers = [TextRankSummarizer(), LexRankSummarizer(), LuhnSummarizer(), LsaSummarizer()]

for i,summarizer in enumerate(summarizers):
    print(summarizer_list[i])
    for sentence in summarizer(parser.document, num_sentences_in_summary):
        print((sentence))
    print("-"*30)