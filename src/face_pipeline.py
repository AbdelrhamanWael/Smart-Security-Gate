from numpy import size
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


##the path of the cascade file which is used to detect the face s is  
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)



# CLAHE normalizes local lighting across face regions much better than global equalizeHist on live webcams
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def crop_face(img_bgr , size=(128 , 128), return_box=False):
    if img_bgr is None: 
        return (None, None) if return_box else None
    

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ## detect the face in the image using the cascade classifier
    ##scale Factor: Parameter specifying how much the image size is reduced at each image scale.
    ##minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
    ##minSize: Minimum possible object size. Objects smaller than that are ignored.
    faces = face_cascade.detectMultiScale(gray , scaleFactor=1.1, minNeighbors=4 , minSize=(40,40))
    if len(faces) == 0:
        return (None, None) if return_box else None

    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])  ##get the largest face detected
    
    # Trim 8% outer border of the Haar box so room background/walls don't pollute HOG/LBP on live webcam
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.08)
    x1 = max(0, x + pad_x)
    y1 = max(0, y + pad_y)
    x2 = min(gray.shape[1], x + w - pad_x)
    y2 = min(gray.shape[0], y + h - pad_y)
    
    cropped = gray[y1:y2 , x1:x2] ##crop the inner face region
    resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA) ##resize the cropped face to the desired size
    enhanced = _CLAHE.apply(resized) ##apply CLAHE adaptive contrast enhancement for webcam lighting invariance

    if return_box:
        return enhanced, (x, y, w, h)
    return enhanced


def hog_feat(face_gray):
    ###capture edge and texture information from the face image using the HOG descriptor
    ## face_gey is the input face image in grayscale format
    ## orientations: Number of orientation bins (0-180 degree) used by the HOG descriptor.
    ## pixels_per_cell: Number of pixels in each cell.
    ## cells_per_block: Number of cells in each block.
    ## block_norm: Normalization method for block descriptors (L2-Hys).
    return hog(face_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

##capture edge and micro-texture information from the face image using the LBP descriptor
def lbp_feat(face_gray , p = 8  ,R=1 , grid=(8,8)):
    ### face_gray is the input face image in grayscale format
    ### p is the number of points used to compute the LBP representation of the face image
    ### R is the radius of the circle used to compute the LBP representation of the face image
    ### grid is the size of the grid used to compute the LBP representation of the face image

    lbp = local_binary_pattern(face_gray, P=p, R=R, method='uniform') 
     ##compute the LBP representation of the face image
     ## uniform: The uniform LBP method is used to compute the LBP representation of the face image. It is a rotation-invariant method that reduces the number of patterns by considering only uniform patterns (patterns with at most two transitions from 0 to 1 or 1 to 0).
    gh = lbp.shape[0] // grid[0]
    gw = lbp.shape[1] // grid[1]
    
    num_bins = p + 2
    histograms = []
    for i in range(grid[0]):## loop over the grid
        for j in range(grid[1]): ## 
            cell = lbp[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            hist, _ = np.histogram(cell, bins=num_bins, range=(0, num_bins), density=True)
            histograms.append(hist)
            
    return np.concatenate(histograms)
def extract_face_features(face_gray, mode="both"):
    """
    The shared interface called by Member 3 for Early Fusion and Hierarchical Mode.
    Combines HOG and LBP into one 2,404-dimensional vector.
    """
    if mode == "hog":
        return hog_feat(face_gray)
    elif mode == "lbp":
        return lbp_feat(face_gray)
    elif mode == "both":
        return np.concatenate([hog_feat(face_gray), lbp_feat(face_gray)])
    else:
        raise ValueError(f"Unknown mode: {mode}")
