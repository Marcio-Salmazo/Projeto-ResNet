from PyQt5.QtCore import pyqtSignal
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

'''
Utilização da rede ResNet pré treinada em razão de:
    * Excelente para extração de características de imagens complexas.
    * Usa "skip connections" (atalhos) para evitar o desaparecimento do gradiente, treinando redes mais profundas.
    * n 
'''


class CNNModel:

    log_signal = pyqtSignal(str)  # sinal para enviar mensagens de log ao PyQt

    def __init__(self, input_shape=(128, 128, 3), num_classes=3):
        """
        Classe responsável por definir a arquitetura de rede neural
        parametro input_shape: Define as dimensões das entradas (imagens).
        parametro num_classes: Define a quantidade de classes na saída.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        # Carregando a ResNet50 sem a camada de saída original
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Definição do modelo como sequêncial, onde as camadas são adicionadas uma após a outra.
        model = models.Sequential([

            base_model,

            # Transforma a saída das camadas convolucionais (um volume 3D) em um vetor 1D
            # para que possa ser passado para a camada densa.
            layers.Flatten(),

            # Definição da primeira camada Densa, totalmente conectada
            # layers.Dense(128, activation='relu'),
            #     * 128 → Quantidade de neurônios na camada
            #     * activation='relu' → Mantém a não linearidade para melhor aprendizado
            #
            # obs: Dropout desativa 50% dos neurônios de forma aleatória a cada iteração, o que evita overfitting,
            # ajudando a rede a generalizar melhor para novos dados

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            # Camada de saída, com número de neurônios igual ao número de classes.
            # A função de ativação softmax transforma os valores de saída em probabilidades para cada classe.
            layers.Dense(self.num_classes, activation='softmax')  # 3 classes (dor, não dor, dor moderada)
        ])

        """
            Função responsável por compilar o modelo e dar inicio ao treinamento
            O metodo .compile() define as configurações do modelo antes do treinamento
                * optimizer='adam' → O Adam (Adaptive Moment Estimation) é um otimizador que
                  ajusta os pesos da rede durante o treinamento de maneira eficiente.
                * loss='categorical_crossentropy' → Essa é a função de erro usada para
                  problemas de classificação multiclasse.
                * metrics=['accuracy'] → Define que a acurácia será monitorada durante o treinamento.
        """
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
