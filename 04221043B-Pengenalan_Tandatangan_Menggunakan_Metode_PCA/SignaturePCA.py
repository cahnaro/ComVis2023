import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Menghasilkan tanda tangan acak untuk tujuan demonstrasi
num_signatures = 100
num_features = 50
signatures = np.random.rand(num_signatures, num_features)

# Pisahkan tanda tangan menjadi set pelatihan dan pengujian
train_signatures = signatures[:80]
test_signatures = signatures[80:]

# Menstandardkan data
scaler = StandardScaler()
train_signatures_scaled = scaler.fit_transform(train_signatures)
test_signatures_scaled = scaler.transform(test_signatures)

# Sesuaikan PCA dengan data pelatihan
pca = PCA(n_components=0.95)  # keep 95% of variance
pca.fit(train_signatures_scaled)

# Pengujian data ke komponen utama
train_pca = pca.transform(train_signatures_scaled)
test_pca = pca.transform(test_signatures_scaled)

# Melatih pengklasifikasi regeresi logistik pada data pelatihan yang diproyeksikan
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_pca, [1 if i < 40 else 0 for i in range(80)])  # 40 pertama adalah tanda tangan asli, 40 terakhir adalah palsu

# Uji pengklasifikasi pada data pengujian yang diproyeksikan
predictions = clf.predict(test_pca)
true_labels = [1 if i < 10 else 0 for i in range(20)]  # 10 pertama adalah tanda tangan asli, 10 terakhir adalah palsu
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)
