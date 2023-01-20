import spacy

# %time 
nlp = spacy.load('es_core_news_md')
# process a sentence using the model
sentence = "Canadá es un país enorme"
print("Frase de ejemplo:", sentence)
mydoc = nlp(sentence)
#Get a vector for individual words
print("Vector para 'Canadá', la primera palabra del texto")
print(mydoc[0].vector) #vector for 'Canada', the first word in the text 
print("Vector de la frase completa - media de los embeddings")
print(mydoc.vector) #Averaged vector for the entire sentence