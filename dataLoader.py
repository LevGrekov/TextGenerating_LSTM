import re
from typing import List

import numpy as np
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences


class DataLoader:
    def __init__(self, file_paths: List[str]):
        self.text = self.__read_files(file_paths)
        self.total_words = 0
        self.tokenizer = None
        self.__tokenize_text()
        self.max_sequence_length = None
        self.dataX = None
        self.dataY = None
        self.__prepare_sequences()

    @staticmethod
    def __read_files(file_paths: List[str]):
        try:
            # Получение текста из нескольких файлов
            texts = []
            for file_path in file_paths:
                with open(file_path, 'r', encoding='windows-1251') as file:
                    text = file.read()
                    texts.append(text)
            combined_text = "\n".join(texts)
            return combined_text
        except Exception as e:
            print(f"An error occurred while reading the files: {e}")
            return None

    def __tokenize_text(self):
        # Токенизация
        text_with_no_latin = re.sub(r'\b\w*[a-zA-Z]+\w*\b', '', self.text)
        # Инициализация токенизатора
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
        # Обучение токенизатора
        tokenizer.fit_on_texts([text_with_no_latin])
        # Получение общего количества слов
        self.total_words = len(tokenizer.word_index) + 1
        self.tokenizer = tokenizer

    def __prepare_sequences(self):
        # Разделение текста на последовательности
        input_sequences = []
        for line in self.text.split("\n"):
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        # Вычисление максимальной длины последовательности, если не задана
        if self.max_sequence_length is None:
            self.max_sequence_length = max(len(seq) for seq in input_sequences)

        # Заполнение нулями и обрезка последовательностей до максимальной длины
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='pre')

        # Разделение входных данных и меток
        x, y = input_sequences[:, :-1], input_sequences[:, -1]
        y = np.expand_dims(y, axis=-1)
        self.dataX = x
        self.dataY = y

    def generate_text(self, seed_text: str, next_words: int, nn_model):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_length - 1, padding='pre')
            predicted_probabilities = nn_model.predict(token_list, verbose=0)

            # Выбор индекса слова с максимальной вероятностью
            predicted = np.argmax(predicted_probabilities)

            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word

        return seed_text
