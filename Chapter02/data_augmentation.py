import os
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

text="The quick brown fox jumps over the lazy dog ."

#OCR Augmentation
aug = nac.OcrAug()  
ocr_augmented_texts = aug.augment(text, n=3) # specifying n=3 gives us only 3 augmented versions of the sentence.

print("Original:")
print(text)

print("OCR Augmented Texts:")
print(ocr_augmented_texts)

#Augmentation at the Word Level
# Downloading the required txt file
import wget

if not os.path.exists("Chapter02/aux_files/spelling_en.txt"):
    wget.download("https://raw.githubusercontent.com/makcedward/nlpaug/5238e0be734841b69651d2043df535d78a8cc594/nlpaug/res/word/spelling/spelling_en.txt", out="Chapter02/aux_files/")
else:
    print("File already exists")

# Substitute word by spelling mistake words dictionary
aug = naw.SpellingAug('Chapter02/aux_files/spelling_en.txt')
word_augmented_texts = aug.augment(text)
print("Word Augmented Texts:")
print(word_augmented_texts)