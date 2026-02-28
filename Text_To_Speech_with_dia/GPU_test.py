import torch

print(torch.cuda.is_available())  # should return True

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


##################


# from PIL import Image
# import pytesseract

# img = Image.open("orv.jpg")
# text = pytesseract.image_to_string(img)
# print(text)
