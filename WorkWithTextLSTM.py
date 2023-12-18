import os
import re
import numpy as np
from keras.src.layers import Embedding, Dropout
from keras.src.saving.saving_api import load_model

import dataLoader as dl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

pushkin = 'data/pushkin_stihi2.txt'
philosophy = ['data/takGovorilZaratustra.txt']
dl = dl.DataLoader([pushkin])

#=============================================================#
#                                                             #
#                       Обычная Модель                        #
#                                                             #
#=============================================================#

s_model = Sequential()
s_model.add(Embedding(dl.total_words, 50, input_length=dl.max_sequence_length - 1))
s_model.add(LSTM(256, return_sequences=True))
s_model.add(LSTM(100))
s_model.add(Dense(dl.total_words, activation='softmax'))

s_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
s_model.summary()
s_model.fit(dl.dataX, dl.dataY, batch_size=32, epochs=1, verbose=1)

#=============================================================#
#                                                             #
#                    Более Сложная Модель                     #
#                                                             #
#=============================================================#
model = Sequential()
model.add(Embedding(dl.total_words, 50, input_length=dl.max_sequence_length - 1))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dense(128, activation='relu'))
model.add(Dense(dl.total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(dl.dataX, dl.dataY, batch_size=32, epochs=1, verbose=1)

#=============================================================#
#                                                             #
#            Уже обученная модель 1000 эпохами                #
#                                                             #
#=============================================================#

# Загрузка модели
loaded_model = load_model("longModel.h5")

max_length = 4
while True:
    seed_text = input("You: ")

    if seed_text.lower() == 'exit':
        print("Выход.")
        break
    if seed_text.isdigit():
        max_length = int(seed_text)
        print(f"max_length = {max_length}")
        continue

    generated_text = dl.generate_text(seed_text, max_length, loaded_model)
    print("NN:", generated_text)

# мой друг в ней и и и и и и и и кляче тиши кляче ней и и и и в цыпочках кляче qui qui mauvaises mauvaises руках крестах mauvaises mauvaises mauvaises mauvaises mauvaises mauvaises mauvaises mauvaises mauvaises mauvaises mauvaises fait fait fait fait fait fait fait qui qui qui qui рыдающая
# мой друг границ евгений и эту добру понимаешь пил глубокий быстротечным пору адских темницу входят направим полуденных столовой полуденных fait полуденных fait кровью отзывы невским записных большого qui порученья fait невским fait fait qui шуму fait невским fait fait невским шуму fait больницы qui крестах больницы qui fait qui qui fait fait
# мой друг в ней бабой бабарихой я и я не ним и fait fait fait в ней бабой бабарихой fait fait fait в fait fait fait fait fait в ней fait fait fait fait fait fait fait fait fait fait fait в ней бабой бабарихой я и fait fait fait fait fait
# "(На второй модели)
# Удалил fait и qui из текста
# мой друг в ним в ним в ним в ним в я я я в я в я в я в я в ним я я я я я я я я я я в ним в ним в ним в ним в ним я я в я в я в я

# 500 эпох (обычная модель):
# мой друг вот бог тебе порука не угадал ей ей познакомить тройка его дивятся наконец и никого сна германии нет балда объяснила мог изабела к докучает брат честном чего и нас милой змия печальна ему горестной забытый озлоблен свыше земная эту возрожденья недавно скажешь в сохнет и дядя всех царь покойна гвидон

# смысл бытия сказал он  – и кто же знаешь еще врачи себя в их душ не есть ли большее наслаждение и возвести в которой есть своя любовь от случая не хочет сжиматься в кулак «существованием» так проповедовало безумие браться на нем – и не хочет быть бережливым ни внезапно праздники и сладость моей воли ибо он должен светить тебя несчастный не может ли проходишь что я не был трубит бы я шатра своего собственного одежд  – но я не хочет видно ослом и не было видно быть истина не было ли проходишь животных и действительно все больше его непрестанно ступающих и наконец
# (20 эпох обучения на "Как говорил заратуштра")
