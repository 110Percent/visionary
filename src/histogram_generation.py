import cv2

from config import config


def generate_histogram(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([grey], [0], None, [256], [0, 256])
    hist = hist.reshape(-1, config.config["histogram"]["bin_size"]).sum(
        axis=1
    )  # Bin by sum into 32 groups of 8 values
    hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist
