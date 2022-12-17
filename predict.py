from Main import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from torchvision import transforms

def make_predictions(model, imagePath):
	model.eval()
	with torch.no_grad():
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image = image.astype("float32")
		image = cv2.resize(image, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))
		orig = image.copy()
		filename = imagePath.split(os.path.sep)[-1]
		image = np.expand_dims(image, 0)
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)
		predMask = model(image).squeeze()
		predMask = predMask.cpu().numpy()
		predMask = predMask * 255
		predMask = predMask.astype(np.uint8)
		pred_file_path = os.path.sep.join([config.PRED_MASK_DIR,filename])
		cv2.imwrite(pred_file_path,predMask)

print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

print("[INFO] load up model...")
maskmodel = torch.load(config.MODEL_PATH).to(config.DEVICE)

# iterate over the randomly selected test image paths
for path in imagePaths:
	make_predictions(maskmodel, path)























