import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Make results reproducible
np.random.seed(42)

# ============================================================
# DATASET
# ============================================================

X = []
y = []

NUM_SAMPLES = 1000
SIGNAL_LENGTH = 20

# ============================================================
# BPSK SIGNALS
# ============================================================

for _ in range(NUM_SAMPLES):

    bits = np.random.randint(0, 2, SIGNAL_LENGTH)

    t = np.linspace(0, 1, SIGNAL_LENGTH)

    carrier = np.cos(2*np.pi*5*t)

    signal = carrier * (2*bits - 1)

    noise = np.random.normal(0, 0.2, SIGNAL_LENGTH)

    noisy_signal = signal + noise

    X.append(noisy_signal)

    y.append(0)

# ============================================================
# BFSK SIGNALS
# ============================================================

for _ in range(NUM_SAMPLES):

    bits = np.random.randint(0, 2, SIGNAL_LENGTH)

    t = np.linspace(0, 1, SIGNAL_LENGTH)

    signal = np.where(
        bits == 0,
        np.sin(2*np.pi*3*t),
        np.sin(2*np.pi*7*t)
    )

    noise = np.random.normal(0, 0.2, SIGNAL_LENGTH)

    noisy_signal = signal + noise

    X.append(noisy_signal)

    y.append(1)

# ============================================================
# QPSK SIGNALS
# ============================================================

for _ in range(NUM_SAMPLES):

    symbols = np.random.randint(0, 4, SIGNAL_LENGTH)

    phases = symbols * (np.pi/2)

    signal = np.cos(phases)

    noise = np.random.normal(0, 0.2, SIGNAL_LENGTH)

    noisy_signal = signal + noise

    X.append(noisy_signal)

    y.append(2)

# ============================================================
# FFT FEATURE EXTRACTION
# ============================================================

features = []

for signal in X:

    fft = np.abs(np.fft.fft(signal))

    feature = np.concatenate((signal, fft))

    features.append(feature)

X = np.array(features)

y = np.array(y)

import matplotlib.pyplot as plt

classes = ["BPSK", "BFSK", "QPSK"]
counts = [sum(y==0), sum(y==1), sum(y==2)]

plt.figure(figsize=(6,4))
plt.bar(classes, counts)
plt.title("Dataset Distribution")
plt.xlabel("Modulation Type")
plt.ylabel("Number of Samples")
plt.grid(True)
plt.show()

print("="*50)
print("Training Samples :", len(X))
print("Classes :", np.unique(y))
print("="*50)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================================
# AI MODEL
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(256,128,64),
    activation="relu",
    solver="adam",
    learning_rate="adaptive",
    max_iter=5000,
    random_state=42
)
print("\nModel Architecture")
print(model)
print("Dataset Ready")
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("="*50)
print("AI BASED MODULATION CLASSIFICATION")
print("="*50)
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("\nModel Architecture")
print(model)
# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining AI Model...")

model.fit(X_train, y_train)
plt.figure(figsize=(8,5))
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

print("\nAI Model Trained Successfully!")

print("AI Model Trained Successfully!")

# ============================================================
# TEST MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy : {:.2f}%".format(accuracy * 100))

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=["BPSK","BFSK","QPSK"]
))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0,1,2]
)

plt.figure(figsize=(6,5))

plt.imshow(cm,cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks([0,1,2],["BPSK","BFSK","QPSK"])

plt.yticks([0,1,2],["BPSK","BFSK","QPSK"])

for i in range(3):
    for j in range(3):
        plt.text(j,i,str(cm[i,j]),
                 ha="center",
                 va="center",
                 color="black")

plt.colorbar()

plt.show()

# ============================================================
# RANDOM SIGNAL TEST
# ============================================================

print("\nRandom Signal Testing")

signal_type = np.random.randint(0,3)

t = np.linspace(0,1,SIGNAL_LENGTH)

if signal_type == 0:

    bits = np.random.randint(0,2,SIGNAL_LENGTH)

    signal = np.cos(2*np.pi*5*t)*(2*bits-1)

    actual = "BPSK"

elif signal_type == 1:

    bits = np.random.randint(0,2,SIGNAL_LENGTH)

    signal = np.where(
        bits==0,
        np.sin(2*np.pi*3*t),
        np.sin(2*np.pi*7*t)
    )

    actual = "BFSK"

else:

    symbols = np.random.randint(0,4,SIGNAL_LENGTH)

    phases = symbols*np.pi/2

    signal = np.cos(phases)

    actual = "QPSK"

noise = np.random.normal(0,0.2,SIGNAL_LENGTH)

test_signal = signal + noise

fft = np.abs(np.fft.fft(test_signal))

test_feature = np.concatenate((test_signal,fft))
prob = model.predict_proba([test_feature])[0]
prediction = model.predict([test_feature])[0]

labels = ["BPSK","BFSK","QPSK"]
print("\nPrediction Confidence")

for i in range(3):
    print(f"{labels[i]} : {prob[i]*100:.2f}%")

labels=["BPSK","BFSK","QPSK"]

print("Actual Signal    :",actual)

print("Predicted Signal :",labels[prediction])

# ============================================================
# TEST SIGNAL GRAPH
# ============================================================

plt.figure(figsize=(10,4))

plt.plot(test_signal)

plt.title("Received Signal")

plt.xlabel("Samples")

plt.ylabel("Amplitude")

plt.grid(True)

plt.show()

# ============================================================
# FFT GRAPH
# ============================================================

plt.figure(figsize=(10,4))

plt.plot(fft)

plt.title("FFT Spectrum")

plt.xlabel("Frequency Bin")

plt.ylabel("Magnitude")

plt.grid(True)

plt.show()
print("\n========== SNR ANALYSIS ==========")

for snr in [20,15,10,5,0]:

    noise_std = 1/(10**(snr/20))

    X_noise=[]

    for signal in X_test:

        noise=np.random.normal(
            0,
            noise_std,
            len(signal)
        )

        X_noise.append(signal+noise)

    X_noise=np.array(X_noise)

    pred=model.predict(X_noise)

    acc=accuracy_score(y_test,pred)

    print(f"SNR = {snr} dB --> Accuracy = {acc*100:.2f}%")

bits = np.random.randint(0,2,200)

I = 2*bits-1

Q = np.zeros(200)

plt.figure(figsize=(6,6))
plt.scatter(I,Q)
plt.title("BPSK Constellation")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid(True)
plt.axis("equal")
plt.show()
# -------------------------
# BPSK Constellation Diagram
# -------------------------

bits = np.random.randint(0, 2, 200)

I = 2 * bits - 1
Q = np.zeros(len(I))

plt.figure(figsize=(6,6))
plt.scatter(I, Q, color='red')
plt.title("BPSK Constellation Diagram")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.axis("equal")
plt.show()
# ============================================================
# QPSK CONSTELLATION
# ============================================================

symbols=np.random.randint(0,4,200)

phase=symbols*np.pi/2

I=np.cos(phase)

Q=np.sin(phase)

plt.figure(figsize=(6,6))

plt.scatter(I,Q)

plt.title("QPSK Constellation")

plt.xlabel("In-Phase")

plt.ylabel("Quadrature")

plt.grid(True)

plt.axis("equal")

plt.show()

# ============================================================
# SNR ANALYSIS
# ============================================================

snr_values=[20,15,10,5,0]

accuracy_list=[]

print("\nSNR Analysis")

for snr in snr_values:

    noise_std=1/(10**(snr/20))

    X_snr=[]

    for sample in X_test:

        noisy=sample+np.random.normal(
            0,
            noise_std,
            len(sample)
        )

        X_snr.append(noisy)

    X_snr=np.array(X_snr)

    pred=model.predict(X_snr)

    acc=accuracy_score(y_test,pred)

    accuracy_list.append(acc*100)

    print(f"SNR={snr} dB  Accuracy={acc*100:.2f}%")

plt.figure(figsize=(8,5))

plt.plot(snr_values,accuracy_list,marker="o")

plt.xlabel("SNR (dB)")

plt.ylabel("Accuracy (%)")

plt.title("Accuracy vs SNR")

plt.grid(True)

plt.show()

print("\nProject Completed Successfully!")