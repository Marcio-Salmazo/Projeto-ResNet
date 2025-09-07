from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QTextEdit, QDialog, QMessageBox, QLabel
)

from CNNModel import CNNModel
from DataParameters import DataParameters
from Model import Model
from NetworkLogName import NetworkLogName
from TrainerThread import TrainerThread
import subprocess
import webbrowser
import time


class Interface(QWidget):

    log_signal = pyqtSignal(str)  # sinal para enviar mensagens de log ao PyQt

    def __init__(self):
        super().__init__()

        self.model = Model()
        self.setWindowTitle("ResNet Trainer")
        self.setWindowIcon(QIcon(self.model.resource_path("figures/figNN")))

        # Layout principal (horizontal), responsável por separar a área que vai mostrar o log de treinamento
        # da seção responsável por conter as funções do programa e configurações da rede
        main_layout = QHBoxLayout(self)

        # Define a área onde o status de treinamento será exibido
        # O QTextEditÉ uma área de texto multilinha que permite a visualização de várias
        # linhas de texto, rolagem automática e até textos formatados (negrito, cores).
        # Ao longo do treinamento da rede, novas mensagens são inseridas aqui.
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)  # somente leitura
        self.log_area.setPlaceholderText("Status do treinamento aparecerá aqui...") # Texto de placeholder
        main_layout.addWidget(self.log_area, stretch=3)  # Adiciona o widget no layout principal

        # Layout vertical para os botões
        button_layout = QVBoxLayout()

        # Definição dos botões
        self.btn_select_folder = QPushButton("Selecionar Dataset")
        self.btn_network_build = QPushButton("Construir ResNet50")
        self.btn_train = QPushButton("Iniciar Treinamento")
        self.btn_tensorboard = QPushButton("Abrir Tensorboard")
        self.btn_exit = QPushButton("Sair")

        # Adiciona botões ao layout vertical
        button_layout.addWidget(self.btn_select_folder)
        button_layout.addWidget(self.btn_network_build)
        button_layout.addWidget(self.btn_train)
        button_layout.addWidget(self.btn_tensorboard)
        button_layout.addWidget(self.btn_exit)
        button_layout.addStretch()  # empurra os botões para cima

        # Adiciona o layout de botões à direita
        main_layout.addLayout(button_layout, stretch=1.5)

        # Conexão dos botões de função
        self.btn_select_folder.clicked.connect(self.select_data)
        self.btn_network_build.clicked.connect(self.build_network)
        self.btn_train.clicked.connect(self.train_network)
        self.btn_tensorboard.clicked.connect(self.open_logs)
        self.btn_exit.clicked.connect(self.exit_program)

        # ----------------------------------------------

        # Atributos referentes ao carregamento dos dados
        self.image_generator_input_size = None
        self.image_generator_batch_size = None
        self.image_generator_split = None
        self.train_data = None
        self.val_data = None

        # ----------------------------------------------

        # Atributos referentes aos parâmetros da ResNet
        self.network_input_size = None
        self.resnet = None
        self.dataset_classes = None

        # ----------------------------------------------

        # Atributos referentes ao treinamento da rede
        self.trainer_thread = None
        self.fileName_weights = None

        # ----------------------------------------------

        # Inserção de label para inserir a logo da UFU
        self.logo_label = QLabel()
        pixmap = QPixmap(self.model.resource_path("figures/fig_ufu.png"))
        pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)  # Centraliza a imagem
        self.logo_label.setContentsMargins(0, 30, 0, 0)  # Padding para espaçar a exibição da imagem
        button_layout.addWidget(self.logo_label)

        # Inserção de label para definir a versão do software
        # Seguindo o padrão de Versionamento Semântico
        # MAJOR.MINOR.PATCH-SUFIX
        self.version_label = QLabel("Ver. 1.0.0-beta", self)
        self.version_label.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(self.version_label)

# ----------------------------------------------------------------------------------------------------------------------

    def add_log_message(self, msg: str):

        self.log_area.append(msg)

    def select_data(self):

        path = self.model.open_directory()
        if path is None:
            QMessageBox.warning(self, "Erro de valor", "Seleção de dados cancelada pelo usuário.")
            return  # encerra a função sem travar

        dialog = DataParameters()
        if dialog.exec_() == QDialog.Accepted:
            self.image_generator_input_size = dialog.input_size
            self.image_generator_batch_size = dialog.batch_size
            self.image_generator_split = dialog.split
        else:
            QMessageBox.warning(self, "Erro de valor","Seleção de dados cancelada pelo usuário.")
            return  # encerra a função sem travar

        self.add_log_message(f'Input Size escolhido: {self.image_generator_input_size}')
        self.add_log_message(f'Batch Size escolhido: {self.image_generator_batch_size}')
        self.add_log_message(f'Taxa de divisão escolhida: {self.image_generator_split}')
        self.add_log_message('--------------------------------------------------------')

        model = Model()

        self.train_data, self.val_data, log_training_samples, log_validation_samples, log_indexes, self.dataset_classes\
            =  (model.load_data(path,
                                (self.image_generator_input_size, self.image_generator_input_size),
                                self.image_generator_batch_size,
                                self.image_generator_split))

        self.add_log_message(log_training_samples)
        self.add_log_message(log_validation_samples)
        self.add_log_message(log_indexes)
        self.add_log_message('--------------------------------------------------------')

    def build_network(self):

        if self.image_generator_input_size is None or self.dataset_classes is None:
            QMessageBox.warning(self, "Erro", "Input Size ou quantidade de classes não foram definidas, recarregue o dataset")
            return

        # O input size da rede tem o mesmo formato dos dados gerados, com 3 camadas (normal de imagens sem tratamento)
        cnn_input_size = (self.image_generator_input_size, self.image_generator_input_size, 3)
        cnn_model = CNNModel(cnn_input_size, self.dataset_classes)
        self.resnet = cnn_model.build_model()

        if self.resnet:
            self.add_log_message(f'Rede construída: {self.resnet}')
            self.add_log_message(f'Rede compilada com sucesso!')
            self.add_log_message(f'Quantidade de classes encontradas: {self.dataset_classes}')
        self.add_log_message('--------------------------------------------------------')

    def train_network(self):

        if self.resnet is None or self.train_data is None or self.val_data is None:
            QMessageBox.warning(self, "Erro", "Dataset inválido para iniciar treinamento ou rede não construída")
            return

        dialog = NetworkLogName()
        if dialog.exec_() == QDialog.Accepted:
            fileName = dialog.log_name
            self.fileName_weights = fileName
            epochs = dialog.epochs
        else:
            QMessageBox.warning(self, "Erro de valor", "Seleção de nome dos logs cancelada pelo usuário.")
            return  # encerra a função sem travar

        # cria a thread de treinamento
        self.trainer_thread = TrainerThread(self.resnet, self.train_data, self.val_data, epochs, fileName)
        self.trainer_thread.log_signal.connect(self.add_log_message)  # conecta o log ao QTextEdit
        self.trainer_thread.training_finished.connect(self.save_weights)  # conecta flag
        self.trainer_thread.start()

    def save_weights(self, success: bool):
        if success:
            self.add_log_message("Treinamento concluído. Salvando pesos:")
            self.resnet.save(f"{self.fileName_weights}_weights.h5")
            self.add_log_message(f"Pesos de treinamento salvos como {self.fileName_weights}_weights.h5")
        else:
            QMessageBox.warning(self, "Erro de valor", "Treinamento não foi iniciado ou concluído")
            return  # encerra a função sem travar

    def open_logs(self):

        log_path = self.model.open_directory()
        if not log_path:
            QMessageBox.warning(self, "Erro", "Caminho para o diretório não foi definido")
            return

        try:
            process = subprocess.Popen(
                ["tensorboard", f"--logdir=\"{log_path}\"", "--port=6006"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(3)  # dá tempo do servidor iniciar
            webbrowser.open("http://localhost:6006")
            self.add_log_message(f'Tensorboard inicializado no caminho: {self.log_path}')
            self.add_log_message('--------------------------------------------------------')

        except ValueError:
            QMessageBox.information(self, 'Erro', 'Erro ao inicializar o tensorboard. Tente novamente')

    def exit_program(self):

        self.close()  # Fecha a janela principal
        QApplication.quit()  # Finaliza o loop da aplicação corretamente
