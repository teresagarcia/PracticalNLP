import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

device = torch.device('cpu')

# text ="""
# don’t build your own MT system if you don’t have to. It is more practical to make use of the translation APIs. When we use such APIs, it is important to pay closer attention to pricing policies. It would perhaps make sense to store the translations of frequently used text (called a translation memory or a translation cache). 

# If you’re working with a entirely new language, or say a new domain where existing translation APIs do poorly, it would make sense to start with a domain knowledge based rule based translation system addressing the restricted scenario you deal with. Another approach to address such data scarce scenarios is to augment your training data by doing “back translation”. Let us say we want to translate from English to Navajo language. English is a popular language for MT, but Navajo is not. We do have a few examples of English-Navajo translation. In such a case, one can build a first MT model between Navajo-English, and use this system to translate a few Navajo sentences into English. At this point, these machine translated Navajo-English pairs can be added as additional training data to English-Navajo MT system. This results in a translation system with more examples to train on (even though some of these examples are synthetic). In general, though, if accuracy of translation is paramount, it would perhaps make sense to form a hybrid MT system which combines the neural models with rules and some form of post-processing, though. 

# """
text = open(f"{os.getcwd()}/data/Chapter05/nlphistory.txt").read()


preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=8,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=3000,
                                    early_stopping=False)
#there are more parameters which can be found at https://huggingface.co/transformers/model_doc/t5.html

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)