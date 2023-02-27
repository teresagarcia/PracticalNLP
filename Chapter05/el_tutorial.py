#From https://github.com/explosion/projects/blob/master/nel-emerson/scripts/notebook_video.ipynb

import os
import csv
from pathlib import Path
import json
import spacy
from spacy.kb import KnowledgeBase
from collections import Counter
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example

print("Load a pretrained English model, apply it to some sample text and show the named entities that were identified")
nlp = spacy.load("en_core_web_lg")
text = "Tennis champion Emerson was expected to win Wimbledon."
print(text)
doc = nlp(text)
for ent in doc.ents:
    print(f"Named Entity '{ent.text}' with label '{ent.label_}'")


print("Creating the Knowledge Base~~~")
print("Load the data from a pre-defined CSV file")
def load_entities():
    entities_loc = Path.cwd() / "data" / "Chapter05" / "entities.csv" 

    names = dict()
    descriptions = dict()
    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions

name_dict, desc_dict = load_entities()
for QID in name_dict.keys():
    print(f"{QID}, name={name_dict[QID]}, desc={desc_dict[QID]}")

print("Create our knowledge base. We define a fixed dimensionality for the entity vectors, 300-D in our case.")

#To add each record to the knowledge base, we encode its description using the built-in word vectors of our nlp model. The vector attribute of a document is the average of its token vectors. We also need to provide a frequency, which is a raw count of how many times a certain entity appears in an annotated corpus. In this tutorial we're not using these frequencies, so we're setting them to an arbitrary value.
qids = name_dict.keys()
def create_kb(vocab):
    kb = KnowledgeBase(vocab, entity_vector_length=300)
    print("Add each record to the knowledge base")
    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)   # 342 is an arbitrary value here

    print("Add alias for name with 100% probability - Roy Emerson is 100% the tennis player")
    for qid, name in name_dict.items():
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])   # 100% prior probability P(entity|alias)

    print("Add alias for Emerson with equal probability for each")
    probs = [0.3 for qid in qids]
    kb.add_alias(alias="Emerson", entities=qids, probabilities=probs)  # sum([probs]) should be <= 1 !
    return kb

kb = create_kb(nlp.vocab) 
print(f"Entities in the KB: {kb.get_entity_strings()}")
print(f"Aliases in the KB: {kb.get_alias_strings()}")

print(f"Candidates for 'Roy Stanley Emerson': {[c.entity_ for c in kb.get_alias_candidates('Roy Stanley Emerson')]}")
print(f"Candidates for 'Emerson': {[c.entity_ for c in kb.get_alias_candidates('Emerson')]}")
print(f"Candidates for 'Sofie': {[c.entity_ for c in kb.get_alias_candidates('Sofie')]}")

print("Save knowledge base and nlp object")
output_dir = Path.cwd()/ "data" / "Chapter05" / "el_output" 
if not os.path.exists(output_dir):
    os.mkdir(output_dir) 
kb.to_disk(output_dir / "my_kb")
nlp.to_disk(output_dir / "my_nlp")

print("Creating a training dataset~~~")
print("We will need some annotated data")

json_loc = Path.cwd() / "data" / "Chapter05" / "emerson_annotated_text.jsonl" # distributed alongside this notebook
with json_loc.open("r", encoding="utf8") as jsonfile:
    line = jsonfile.readline()
    print(line)   # print just the first line


print("Training the Entity Linker~~~")
print("To feed training data into our Entity Linker, we need to format our data as a structured tuple.")

dataset = []
with json_loc.open("r", encoding="utf8") as jsonfile:
    for line in jsonfile:
        example = json.loads(line)
        text = example["text"]
        if example["answer"] == "accept":
            QID = example["accept"][0]
            offset = (example["spans"][0]["start"], example["spans"][0]["end"])
            links_dict = {QID: 1.0}
        dataset.append((text, {"links": {offset: links_dict}}))

print(dataset[3])

print("How many cases of each QID do we have annotated?")
gold_ids = []
for text, annot in dataset:
    for span, links_dict in annot["links"].items():
        for link, value in links_dict.items():
            if value:
                gold_ids.append(link)

print(Counter(gold_ids))

print("Split training and test dataset")
train_dataset = []
test_dataset = []
for QID in qids:
    indices = [i for i, j in enumerate(gold_ids) if j == QID]
    train_dataset.extend(dataset[index] for index in indices[0:8])  # first 8 in training
    test_dataset.extend(dataset[index] for index in indices[8:10])  # last 2 in test
    
random.shuffle(train_dataset)
random.shuffle(test_dataset)

print("We'll first run each of our training sentences through the pipeline with the NER component")
TRAIN_DOCS = []
for text, annotation in train_dataset:
    doc = nlp(text)     # to make this more efficient, you can use nlp.pipe() just once for all the texts
    TRAIN_DOCS.append((doc, annotation))

print("Then, we'll create a new Entity Linking component and add it to the pipeline.")
entity_linker = nlp.create_pipe("entity_linker", config={"incl_prior": False})
entity_linker.set_kb(create_kb)
nlp.add_pipe('entity_linker', last=True)

print("Run the actual training loop for the new component, taking care to only train the entity linker and not the other components.")

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]
with nlp.disable_pipes(*other_pipes):   # train only the entity_linker
    optimizer = nlp.create_optimizer()
    for itn in range(500):   # 500 iterations takes about a minute to train
        random.shuffle(TRAIN_DOCS)
        batches = minibatch(TRAIN_DOCS, size=compounding(4.0, 32.0, 1.001))  # increasing batch sizes
        losses = {}
        for batch in batches:
            for text, annotations in batch:
                # create Example
                # doc = nlp.make_doc(text)
                example = Example.from_dict(text, annotations)
                nlp.update(
                    [example],  
                    drop=0.2,      # prevent overfitting
                    losses=losses,
                    sgd=optimizer,
                )
        if itn % 50 == 0:
            print(itn, "Losses", losses)   # print the training loss
print(itn, "Losses", losses)
