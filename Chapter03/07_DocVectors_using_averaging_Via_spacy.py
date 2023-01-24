#Import spacy and load the model
import spacy
nlp = spacy.load("en_core_web_sm") #here nlp object refers to the 'en_core_web_sm' language model instance.

#Assume each sentence in documents corresponds to a separate document.
documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs

print("Document After Pre-Processing:",processed_docs)


#Iterate over each document and initiate an nlp instance.
for doc in processed_docs:
    doc_nlp = nlp(doc) #creating a spacy "Doc" object which is a container for accessing linguistic annotations. 
    
    print("-"*30)
    print(f"""Average Vector of '{doc}'\n""",doc_nlp.vector)#this gives the average vector of each document
    for token in doc_nlp:
        print()
        print(token.text,token.vector)#this gives the text of each word in the doc and their respective vectors.