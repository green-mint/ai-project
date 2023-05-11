import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import itertools
import math
import time


class BiGRU(nn.Module):
    def __init__(self, total_classes):
        super(BiGRU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.gru1 = nn.GRU(256, 256, bidirectional=True)
        self.gru1_bn1 = nn.BatchNorm1d(512)
        self.gru2 = nn.GRU(512, 512, bidirectional=True)
        self.gru2_bn2 = nn.BatchNorm1d(1024)
        self.classifier = nn.Conv1d(1024, total_classes + 1, kernel_size=1)

    def forward(self, x):  # (*, 1, 64, 1300)
        x = self.conv1(x)  # (*, 16, 32, 650)
        x = self.conv2(x)  # (*, 32, 16, 325)
        x = self.conv3(x)  # (*, 64, 8, 325)
        x = self.conv4(x)  # (*, 128, 4, 325)
        x = self.conv5(x)  # (*, 256, 1, 322)
        x = x.squeeze(2)
        x, _ = self.gru1(x.permute(2, 0, 1))  # (322, *, 512)
        x = F.leaky_relu(self.gru1_bn1(x.permute(1, 2, 0)))  # (*, 512, 322)
        x, _ = self.gru2(x.permute(2, 0, 1))  # (322, *, 1024)
        x = F.leaky_relu(self.gru2_bn2(x.permute(1, 2, 0)))  # (*, 1024, 322)
        return F.log_softmax(self.classifier(x), dim=1)  # (*, 1024, 320)


def resize_image(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized


def load_characters(character_file):
    characters = []
    with open(character_file, "r", encoding="utf-8") as file:
        for line in file:
            characters.append(line.replace("\n", ""))
    return characters


def load_model(path, total_characters, device='mps'):
    model = BiGRU(total_characters)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def preprocess(img, max_height=64, max_width=1300):
    height, width = img.shape[:2]
    if height != max_height:
        img = resize_image(img, height=max_height)
    img = cv.flip(img, 1)
    height, width = img.shape[:2]
    # if width > max_width:
    #     width = max_width
    img = img.astype(np.float32)
    img = (img / 255.0) * 2.0 - 1.0
    array = np.zeros((max_height, max_width))
    width = min(width, max_width)
    array[:, :width] = img[:, :width]
    array = array.T
    array = np.expand_dims(array, -1)
    return array


def process_tensor(model, tensor, characters, width=None):
    with torch.no_grad():
        scores = model(tensor).permute(0, 2, 1)
        # print(scores)
        indices = torch.argmax(scores, dim=2).cpu().numpy()
        # print(indices.shape)
    output = []
    for i, x in enumerate(indices):
        if width is not None:
            x = x[:width[i] // 4 - 3 + 16]
        text = []
        x = [k for k, g in itertools.groupby(list(x))]
        for c in x:
            if c != 0 and c < len(characters):
                text.append(characters[c - 1])
        output.append("".join(text))
    return output


def process_images(model, images, characters, device='mps', lock=None):
    with torch.no_grad():
        images = [resize_image(image, height=64) for image in images]
        width = [image.shape[1] for image in images]
        tensor = torch.stack([torch.FloatTensor(preprocess(image)).permute(2, 1, 0) for image in images])
        if device is not None:
            tensor = tensor.to(device, non_blocking=True)
        if lock is None:
            labels = process_tensor(model, tensor, characters, width)
        else:
            lock.acquire()
            labels = process_tensor(model, tensor, characters, width)
            lock.release()
        return labels


def process_images_safe(model, images, characters, device='mps', batch_size=16, lock=None):
    labels = []
    for i in range(math.ceil(len(images) / batch_size)):
        batch = images[i * batch_size: (i + 1) * batch_size]
        labels.extend(process_images(model, batch, characters, device, lock))
    return labels

def predict_image(model, characters, image_path, dims, device='cpu', batch_size=16):
    img = cv.imread(image_path, 0)
    print(img.shape)
    # crop the image at dims(x, y, w, h)
    if dims[3] != 0 and dims[2] != 0:
        img = img[dims[1]:dims[1] + dims[3], dims[0]:dims[0] + dims[2]]
    
    print(img.shape)
    # write the image
    cv.imwrite("test.jpg", img)
    labels = process_images_safe(model, [img], characters, device, batch_size)
    return labels[0]


if __name__ == "__main__":
    import glob
    import argparse
    device = 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default="./Weights/urdu.model", type=str, help="model weights file")
    parser.add_argument("-chars", default="./Weights/chars.txt", type=str, help="model characters file")
    parser.add_argument("-input", default="./Images/", type=str, help="OCR input directory")
    parser.add_argument("-batch", default=256, type=int, help="batch size")
    args = parser.parse_args()

    characters = load_characters(args.chars)
    model = load_model(args.model, len(characters), device).to(device)
    model.eval()
    files = glob.glob(args.input + "*.jpg")
    print(files)
    images = [cv.imread(file, 0) for file in files]
    print(len(images))
    start_time = time.perf_counter()
    labels = process_images_safe(model, images, characters, device, args.batch)
    end_time = time.perf_counter()
    for label in labels:
        print(label)
    print("Time taken to process is " + str(end_time - start_time) + " seconds")