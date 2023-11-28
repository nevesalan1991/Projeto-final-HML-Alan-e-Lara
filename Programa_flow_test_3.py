#!/usr/bin/env python
# coding: utf-8

# In[75]:


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import pandas as pd
#from scipy.interpolate import interp2d
#from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator

# In[76]:

########################################################################################################################################
#######################   1) Pré-processamento   #########################################################################
######################################################################################################################
# Crie listas para armazenar os vetores de espectrograma e as classes
vetores_espectrograma = []
classes = []



from scipy.interpolate import RegularGridInterpolator

def padronizar_espectrograma(Sxx):
    novo_shape = (343, 129)  # Defina o novo shape desejado

    # Crie uma grade regular para as coordenadas x e y
    x_vals = np.linspace(0, Sxx.shape[1] - 1, Sxx.shape[1])
    y_vals = np.linspace(0, Sxx.shape[0] - 1, Sxx.shape[0])

    interp_func = RegularGridInterpolator((y_vals, x_vals), Sxx, method='linear', bounds_error=False, fill_value=None)

    # Crie uma nova grade de coordenadas
    new_y_vals = np.linspace(0, Sxx.shape[0] - 1, novo_shape[0])
    new_x_vals = np.linspace(0, Sxx.shape[1] - 1, novo_shape[1])

    # Crie a grade completa de coordenadas para avaliar a função interpoladora
    coords = np.array(np.meshgrid(new_y_vals, new_x_vals, indexing='ij')).T.reshape(-1, 2)

    # Avalie a função interpoladora na nova grade de coordenadas
    Sxx_padronizado = interp_func(coords).reshape(novo_shape)

    return Sxx_padronizado


def read_mat_files_and_save(folder_path, save_folder):
    global vetores_espectrograma, classes

    os.makedirs(save_folder, exist_ok=True)
    #folder_class = len(set(classes)) + 1
    folder_class = len(set(classes)) + 1 if 'folder_class' not in locals() else folder_class
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    for mat_file in mat_files:
        file_path = os.path.join(folder_path, mat_file)
        mat_data = loadmat(file_path)
        terceira_coluna = mat_data['dados_xlsx'][:, 2]

        fs = 1000
        nperseg = 256
        noverlap = nperseg // 2

        f, t, Sxx = signal.spectrogram(terceira_coluna, fs, nperseg=nperseg, noverlap=noverlap)

        # Padronizar o espectrograma
        Sxx_padronizado = padronizar_espectrograma(Sxx)

        # print("Dimensões de t:", t.shape)
        # print("Dimensões de f:", f.shape)
        # print("Dimensões de Sxx_padronizado:", Sxx_padronizado.shape)

        # Antes de chamar plt.pcolormesh, ajuste as dimensões de t e f
        t = np.linspace(0, Sxx.shape[1] - 1, Sxx.shape[1])  # Certifique-se de ajustar o intervalo conforme necessário
        f = np.linspace(0, Sxx.shape[0] - 1, Sxx.shape[0])  # Certifique-se de ajustar o intervalo conforme necessário

        # Certifique-se de que t e f tenham uma dimensão a menos do que Sxx_padronizado
        t, f = np.meshgrid(t, f, indexing='ij')


        # Plotagem do espectrograma
        plt.figure()
        plt.pcolormesh(t, f, 10 * np.log10(Sxx_padronizado), shading='auto')
        plt.colorbar(label='Potência (dB)')
        plt.ylabel('Frequência (Hz)')
        plt.xlabel('Tempo (s)')
        plt.title(f'Espectrograma - Classe {folder_class}')

        # Salva o espectrograma plotado como imagem
        save_path = os.path.join(save_folder, f"espectrograma_{mat_file.replace('.mat', '.png')}")
        plt.savefig(save_path)
        plt.close()

        # Adiciona às listas globais
        #vetores_espectrograma.append(Sxx_padronizado.flatten())
        vetores_espectrograma.append(Sxx.ravel())
        #classes.append(folder_class)
        
        # Calcula estatísticas descritivas no Sxx_padronizado
        mean_value = np.mean(Sxx_padronizado)
        std_dev = np.std(Sxx_padronizado)
        max_value = np.max(Sxx_padronizado)
        min_value = np.min(Sxx_padronizado)


        # Adiciona as estatísticas descritivas às listas globais
        estatisticas_descritivas = [mean_value, std_dev, max_value, min_value]
        #vetores_espectrograma.append(estatisticas_descritivas)
        classes.append(folder_class)

    print(f"Processed folder: {folder_path}, Class: {folder_class}")
    # print(f"Class: {folder_class}")

# In[77]:


# Dicionário de pastas com seus respectivos caminhos
folder_paths = {
    'Estrat_Liso_7nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Estrat_Liso_7nm3h',
    'Estrat_ond_16nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Estrat_ond_16nm3h',
    'Estrat_ond_24nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Estrat_ond_24nm3h',
    'Plug_5nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Plug_5nm3h',
    'Plug_10nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Plug_10nm3h',
    'Slug_15nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Slug_15nm3h',
    'Slug_45nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Slug_45nm3h',
    'wavy--slug_55nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/wavy--slug_55nm3h',
}


# In[78]:


# Iterar sobre o dicionário e chamar a função para cada pasta
for folder_name, folder_path in folder_paths.items():
    print(f"Processing folder: {folder_name}")
    save_folder = os.path.join('C:\\Users\\larac\\OneDrive - puc-rio.br\\DOUTORADO PUC-Rio\\2_Periodo\\Hands-on Machine Learning\\Projeto Final\\Flow Pattern Classification\\dados_mat\\Espectograma', folder_name)
    read_mat_files_and_save(folder_path, save_folder)

##############################################################################################################################################################################################################################
#######################  2) Extração dos dados ############################################################################
###################################################################################################################################
vetores_espectrograma = [arr.flatten() if isinstance(arr, np.ndarray) and len(arr.shape) > 1 else arr for arr in vetores_espectrograma if not isinstance(arr, list)]

# Removendo elementos não-array da lista
vetores_espectrograma = [arr for arr in vetores_espectrograma if isinstance(arr, np.ndarray)]

# Converter a lista de vetores e classes em matrizes
matriz_espectrograma = np.array(vetores_espectrograma)
vetor_classes = np.array(classes)

# Imprima a forma das matrizes
print('Matriz de Espectrograma:', matriz_espectrograma.shape)
print('Vetor de Classes:', vetor_classes.shape)


# In[83]:

########################################################################################################################################
####################### 3) Redução de Dimensionalidade ################################################################################
#######################################################################################################################################

# Aplicar PCA na Matriz de Espectrograma
num_componentes = 0.95  # Especifica diretamente o número de componentes
pca = PCA(n_components=num_componentes)
matriz_reduzida = pca.fit_transform(matriz_espectrograma)

# Imprimir a nova forma da matriz após a redução
print(f'Matriz Reduzida após PCA ({num_componentes} componentes):', matriz_reduzida.shape)


# In[84]:


# Criar um DataFrame para plotar o gráfico dispersão
pca_df = pd.DataFrame(matriz_reduzida, columns=[f'Componente Principal {i + 1}' for i in range(matriz_reduzida.shape[1])])
pca_df['Classe'] = np.repeat([folder_name for folder_name in folder_paths.keys()], [len(os.listdir(folder_paths[folder_name])) for folder_name in folder_paths.keys()])



# In[85]:


plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='Componente Principal 1', y='Componente Principal 2', hue='Classe', palette='viridis', s=80)
plt.title('PCA - Gráfico de Dispersão com Legendas')


# In[86]:


save_pca_folder = 'C:\\Users\\larac\\OneDrive - puc-rio.br\\DOUTORADO PUC-Rio\\2_Periodo\\Hands-on Machine Learning\\Projeto Final\\Flow Pattern Classification\\dados_mat\\PCA_Graficos'
os.makedirs(save_pca_folder, exist_ok=True)
plt.savefig(os.path.join(save_pca_folder, 'pca_plot.png'))

############################################################################################################################################################################
########################## 4) Divisão dos dados ###################################################
########################################################################################################

X_train, X_test, y_train, y_test = train_test_split(matriz_reduzida, vetor_classes, test_size=0.4, random_state=42)


###################################################################################################################################
########################## 5) Treinamento de Classificação  ##############################################################
##################################################################################################################################

# In[]:

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)

# Avaliar o desempenho do KNN
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)
print("Classification Report (KNN):\n", classification_report(y_test, knn_predictions))

# SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

# Avaliar o desempenho do SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)
print("Classification Report (SVM):\n", classification_report(y_test, svm_predictions))

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

# Avaliar o desempenho do Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
print("Classification Report (Random Forest):\n", classification_report(y_test, rf_predictions))

# In[]:

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# KNN Confusion Matrix
sns.heatmap(confusion_matrix(y_test, knn_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.show()

# SVM Confusion Matrix
sns.heatmap(confusion_matrix(y_test, svm_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.show()

# Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.show()

# In[]:

# Dados reais dos modelos
modelos = ['KNN', 'SVM', 'Random Forest']
acuracias = [knn_accuracy, svm_accuracy, rf_accuracy]

# Criar um DataFrame usando Pandas
data = {'Modelo': modelos, 'Acurácia': acuracias}
df_acuracia = pd.DataFrame(data)

# Plotar a tabela usando Matplotlib
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df_acuracia.values, colLabels=df_acuracia.columns, cellLoc='center', loc='center')

plt.show()

####################################################################################################################
#################### 6) Avaliação do Modelo #######################################################################
###################################################################################################################
'''
Testes dos Hiperparâmetros
'''

# In[]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Definir os hiperparâmetros e os valores a serem testados para o KNN
parametros_knn = {'n_neighbors': [3, 5, 7, 9],
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan']}

# Criar o modelo KNN
knn = KNeighborsClassifier()

# Criar o objeto GridSearchCV para o KNN
grid_search_knn = GridSearchCV(knn, parametros_knn, cv=5, scoring='accuracy')

# Executar a busca em grade na matriz reduzida
grid_search_knn.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros encontrados para o KNN
print("Melhores Hiperparâmetros para KNN:", grid_search_knn.best_params_)

# Imprimir a melhor precisão encontrada para o KNN
print("Melhor Precisão para KNN:", grid_search_knn.best_score_)


# In[]:

from sklearn.svm import SVC

# Definir os hiperparâmetros e os valores a serem testados para a SVM
parametros_svm = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  'gamma': ['scale', 'auto']}

# Criar o modelo SVM
svm = SVC()

# Criar o objeto GridSearchCV para a SVM
grid_search_svm = GridSearchCV(svm, parametros_svm, cv=5, scoring='accuracy')

# Executar a busca em grade na matriz reduzida
grid_search_svm.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros encontrados para a SVM
print("Melhores Hiperparâmetros para SVM:", grid_search_svm.best_params_)

# Imprimir a melhor precisão encontrada para a SVM
print("Melhor Precisão para SVM:", grid_search_svm.best_score_)


# In[]:


from sklearn.ensemble import RandomForestClassifier

# Definir os hiperparâmetros e os valores a serem testados para o Random Forest
parametros_rf = {'n_estimators': [50, 100, 200],
                 'max_depth': [None, 10, 20, 30],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}

# Criar o modelo Random Forest
rf = RandomForestClassifier(random_state=42)

# Criar o objeto GridSearchCV para o Random Forest
grid_search_rf = GridSearchCV(rf, parametros_rf, cv=5, scoring='accuracy')

# Executar a busca em grade na matriz reduzida
grid_search_rf.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros encontrados para o Random Forest
print("Melhores Hiperparâmetros para Random Forest:", grid_search_rf.best_params_)

# Imprimir a melhor precisão encontrada para o Random Forest
print("Melhor Precisão para Random Forest:", grid_search_rf.best_score_)


# In[]:


# Melhores hiperparâmetros e precisão para cada modelo
melhores_resultados = {
    'Modelo': ['KNN', 'SVM', 'Random Forest'],
    'Melhores Hiperparâmetros': [grid_search_knn.best_params_, grid_search_svm.best_params_, grid_search_rf.best_params_],
    'Melhor Precisão': [ grid_search_knn.best_score_, grid_search_svm.best_score_, grid_search_rf.best_score_]
}

# Criar DataFrame com os resultados
df_resultados = pd.DataFrame(melhores_resultados)

# Exibir a tabela
print(df_resultados)
