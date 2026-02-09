import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from src.config import N_FEATURES_ORB, VOCAB_SIZE, PCA_COMPONENTS, RANDOM_STATE

class VisualFeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=N_FEATURES_ORB)
        self.kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=1000, random_state=RANDOM_STATE)
        self.pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        self.is_fitted = False

    def extract_orb_descriptors(self, images):
        all_descriptors = []
        image_descriptors = []
        
        for img in images:
            img_uint8 = (img * 255).astype(np.uint8)
            _, descriptors = self.orb.detectAndCompute(img_uint8, None)
            
            if descriptors is not None:
                image_descriptors.append(descriptors)
                all_descriptors.append(descriptors)
            else:
                image_descriptors.append(np.empty((0, 32)))
                
        if all_descriptors:
            all_descriptors = np.vstack(all_descriptors)
            
        return all_descriptors, image_descriptors

    def create_bovw_histograms(self, image_descriptors):
        bovw_features = []
        for descriptors in image_descriptors:
            hist = np.zeros(VOCAB_SIZE)
            if descriptors.shape[0] > 0:
                cluster_labels = self.kmeans.predict(descriptors)
                for label in cluster_labels:
                    hist[label] += 1
            # Normalize
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            bovw_features.append(hist)
        return np.array(bovw_features)

    def fit_transform(self, images):
        print("Extracting ORB descriptors...")
        all_desc, img_desc = self.extract_orb_descriptors(images)
        
        print(f"Building Visual Vocabulary (K={VOCAB_SIZE})...")
        self.kmeans.fit(all_desc)
        
        print("Creating BoVW Histograms...")
        X_bovw = self.create_bovw_histograms(img_desc)
        
        print(f"Reducing dimensionality with PCA ({PCA_COMPONENTS})...")
        X_reduced = self.pca.fit_transform(X_bovw)
        
        self.is_fitted = True
        return X_reduced

    def transform(self, images):
        if not self.is_fitted:
            raise Exception("Extractor not fitted")
        _, img_desc = self.extract_orb_descriptors(images)
        X_bovw = self.create_bovw_histograms(img_desc)
        return self.pca.transform(X_bovw)