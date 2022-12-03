import torch
import clip
import os
from PIL import Image
from scipy.spatial import distance
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


img_dir = 'F:/AvanPost/Dataset/8'
img_file_name = '2138.jpg'

list_image_files = [item for item in os.listdir(img_dir)]
duplicates_list = []

embeddings = []

# image = preprocess(Image.open(f"{img_dir}/{img_file_name}")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)


thresh = 0.05

with torch.no_grad():
    for file in tqdm(list_image_files):
        image = preprocess(Image.open(f"{img_dir}/{file}")).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        embeddings.append(image_features.cpu().numpy())
        if len(embeddings) > 400:
            break

for i, emb in enumerate(embeddings):
    for j, item in enumerate(embeddings):
        if i == j:
            continue
        dist = distance.cosine(emb, item)
        if dist < thresh:
            duplicates_list.append((list_image_files[i], list_image_files[j]))
            # print(f'{list_image_files[i]} is close to {list_image_files[j]}')


print('duplicated pairs: \n')
print(set([tuple(sorted(i)) for i in duplicates_list]))


    # image_features = model.encode_image(image)
    # print(image_features.shape)
    # text_features = model.encode_text(text)

    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]