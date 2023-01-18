import pytesseract
from pytesseract import image_to_string
from PIL import Image

image_path=r"imgs/unita_ejemplo.jpg"
extractedInformation = image_to_string(Image.open(image_path))
print(extractedInformation)