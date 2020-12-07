import numpy as np
from matplotlib.image import imread
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import cv2
np.seterr(divide='ignore', invalid='ignore')

# leitura dos arquvos de imagens
frame1 = imread(r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\filtered_mask\image290.png') # imagem predita
frame2 = imread(r'C:\Users\cassio\PycharmProjects\FinalProject\aton_lab\shadows\aton_lab_0291.png') # imagem referência

cutoff = 0.49804    # valor de corte para as sombras

y_pred_classes = np.zeros_like(frame1, dtype=int)   # matriz de valores pretidos
y_test_classes = np.zeros_like(frame2, dtype=int)   # matriz de valores detectados

for i in range(len(frame1)):
    for j in range(len(frame1[i])):
        if frame1[i][j] > cutoff:
            y_pred_classes[i][j] = 1

for i in range(len(frame2)):
    for j in range(len(frame2[i])):
        if frame2[i][j] > cutoff:
            y_test_classes[i][j] = 1

cm = confusion_matrix(y_test_classes.argmax(axis=1), y_pred_classes.argmax(axis=1)) #matriz de confusão

precision = precision_score(y_test_classes.argmax(axis=1), y_pred_classes.argmax(axis=1), average='micro')  # precisão
recall = recall_score(y_test_classes.argmax(axis=1), y_pred_classes.argmax(axis=1), average='micro') # revocação
f1 = f1_score(y_test_classes.argmax(axis=1), y_pred_classes.argmax(axis=1), average='micro') # medida F
accuracy = accuracy_score(y_test_classes.argmax(axis=1), y_pred_classes.argmax(axis=1))

print(precision)
print(recall)
print(f1)


df_cm = pd.DataFrame(cm, range(21), range(21))
sn.set(font_scale=1.0)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size

plt.show()

