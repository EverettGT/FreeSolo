"""
1D-CNN Feature Extractor + MLP Classifier

Binary classification: 0=Human (eject), 1=Pathogen (keep)
"""
import numpy as np
import os
import json
import time
import pickle
from scipy.stats import skew, kurtosis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)


#CNN feature extractor
class Conv1DFeatureExtractor:
    def __init__(self):
        self.filters = self._build_filter_bank()
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _build_filter_bank(self):
        filters = []
        for width in [5, 10, 20, 40]:
            f = np.zeros(width); f[:width//2] = -1; f[width//2:] = 1
            filters.append(f / np.linalg.norm(f))
        for sigma in [3, 7, 15]:
            x = np.arange(-3*sigma, 3*sigma+1)
            f = -x * np.exp(-x**2 / (2*sigma**2))
            filters.append(f / np.linalg.norm(f))
        for freq in [0.01, 0.02, 0.05, 0.1, 0.2]:
            t = np.arange(50); f = np.sin(2*np.pi*freq*t)
            filters.append(f / np.linalg.norm(f))
        for width in [5, 15, 50]:
            filters.append(np.ones(width) / width)
        return filters

    def _apply_convolutions(self, signal_1d):
        features = []
        for filt in self.filters:
            conv_out = np.convolve(signal_1d, filt, mode='valid')
            relu_out = np.maximum(0, conv_out)
            n_pools = 10
            pool_size = len(relu_out) // n_pools
            pooled = [np.max(relu_out[i*pool_size:(i+1)*pool_size])
                     for i in range(n_pools)] if pool_size > 0 else [np.max(relu_out)]
            features.extend([np.mean(relu_out), np.std(relu_out),
                           np.max(relu_out), np.median(relu_out)])
            features.extend(pooled)
        return features

    def _extract_statistical_features(self, signal_1d):
        features = [np.mean(signal_1d), np.std(signal_1d), np.median(signal_1d),
                   skew(signal_1d), kurtosis(signal_1d), np.min(signal_1d),
                   np.max(signal_1d), np.ptp(signal_1d)]
        n_seg = 20; seg_len = len(signal_1d) // n_seg
        for i in range(n_seg):
            seg = signal_1d[i*seg_len:(i+1)*seg_len]
            features.extend([np.mean(seg), np.std(seg)])
        diff1 = np.diff(signal_1d); diff2 = np.diff(diff1)
        features.extend([np.mean(np.abs(diff1)), np.std(diff1),
                        np.mean(np.abs(diff2)), np.std(diff2)])
        centered = signal_1d - np.mean(signal_1d)
        features.append(np.sum(np.abs(np.diff(np.sign(centered)))) / (2*len(centered)))
        fft_vals = np.abs(np.fft.rfft(signal_1d))
        fft_freqs = np.fft.rfftfreq(len(signal_1d))
        features.append(np.sum(fft_freqs*fft_vals) / (np.sum(fft_vals)+1e-10))
        n_bands = 10; band_size = len(fft_vals) // n_bands
        for i in range(n_bands):
            features.append(np.sum(fft_vals[i*band_size:(i+1)*band_size]**2))
        autocorr = np.correlate(centered[:500], centered[:500], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        for lag in [1, 5, 10, 25, 50, 100]:
            features.append(autocorr[lag] if lag < len(autocorr) else 0.0)
        return features

    def extract_features(self, X):
        print(f"Extracting features from {X.shape[0]} signals...")
        all_features = []
        for i in range(X.shape[0]):
            conv_feats = self._apply_convolutions(X[i])
            stat_feats = self._extract_statistical_features(X[i])
            all_features.append(conv_feats + stat_feats)
            if (i+1) % 1000 == 0: print(f"  {i+1}/{X.shape[0]}")
        features = np.nan_to_num(np.array(all_features, dtype=np.float64),
                                nan=0.0, posinf=1e10, neginf=-1e10)
        return features

    def fit_transform(self, X):
        features = self.extract_features(X)
        features = self.scaler.fit_transform(features)
        self._is_fitted = True
        return features

    def transform(self, X):
        return self.scaler.transform(self.extract_features(X))

#Binary classifier
class VSIClassifier:
    def __init__(self, hidden_layers=(256, 128, 64), max_iter=500, random_state=42):
        self.feature_extractor = Conv1DFeatureExtractor()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers, activation='relu', solver='adam',
            alpha=1e-4, batch_size=64, learning_rate='adaptive',
            learning_rate_init=1e-3, max_iter=max_iter, random_state=random_state,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, verbose=True)
        self.training_history = {}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print("\n" + "="*60 + "\nVSI Classifier Training\n" + "="*60)
        t0 = time.time()
        X_train_feat = self.feature_extractor.fit_transform(X_train)
        feat_time = time.time() - t0
        print(f"  Features: {X_train_feat.shape}, time: {feat_time:.1f}s")
        if X_val is not None:
            X_val_feat = self.feature_extractor.transform(X_val)
        t0 = time.time()
        self.model.fit(X_train_feat, y_train)
        train_time = time.time() - t0
        self.training_history = {
            'loss_curve': self.model.loss_curve_, 'n_iterations': self.model.n_iter_,
            'feature_extraction_time': feat_time, 'training_time': train_time,
            'n_features': X_train_feat.shape[1], 'n_train_samples': len(y_train),
            'train_accuracy': accuracy_score(y_train, self.model.predict(X_train_feat)),
        }
        if X_val is not None:
            self.training_history['val_accuracy'] = accuracy_score(
                y_val, self.model.predict(X_val_feat))
        print(f"  Train acc: {self.training_history['train_accuracy']:.4f}")
        return self.training_history

    def predict(self, X):
        return self.model.predict(self.feature_extractor.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.feature_extractor.transform(X))

    def predict_single(self, signal):
        t0 = time.perf_counter()
        signal = signal.reshape(1, -1)
        proba = self.model.predict_proba(self.feature_extractor.transform(signal))[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        pred = int(np.argmax(proba))
        return pred, float(proba[pred]), latency_ms

    def evaluate(self, X_test, y_test):
        print("\n" + "="*60 + "\nVSI Classifier Evaluation\n" + "="*60)
        features = self.feature_extractor.transform(X_test)
        y_pred = self.model.predict(features)
        y_proba = self.model.predict_proba(features)[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(
                y_test, y_pred, target_names=['Human', 'Pathogen']),
        }
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        print("\nBenchmarking latency...")
        latencies = []
        for i in range(min(100, len(X_test))):
            _, _, lat = self.predict_single(X_test[i])
            latencies.append(lat)
        metrics['latency'] = {
            'mean_ms': float(np.mean(latencies)), 'std_ms': float(np.std(latencies)),
            'median_ms': float(np.median(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'min_ms': float(np.min(latencies)), 'max_ms': float(np.max(latencies)),
        }
        print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  Latency:   {metrics['latency']['mean_ms']:.2f} ms (mean)")
        print(f"\n{metrics['classification_report']}")
        return metrics

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'feature_extractor': self.feature_extractor,
                        'training_history': self.training_history}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_extractor = data['feature_extractor']
        self.training_history = data['training_history']
