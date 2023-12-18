import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.src.layers import Dropout

# Создание примерного временного ряда
np.random.seed(0)
timesteps = 190
x_values = np.arange(timesteps)
y_values = np.sin(0.1 * x_values) + 0.1 * np.random.randn(timesteps)

# Разделение данных на обучающую и тестовую выборки
train_size = int(timesteps * 0.8)
train_data, test_data = y_values[:train_size], y_values[train_size:]

# Функция для создания последовательных входных и выходных данных для модели
def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# Задаем количество временных шагов для предсказания будущих значений
n_steps = 10

# Создание обучающего и тестового наборов данных
X_train, y_train = create_dataset(train_data, n_steps)
X_test, y_test = create_dataset(test_data, n_steps)


model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Тренировка модели
model.fit(X_train, y_train, epochs=1000, batch_size=16, verbose=1)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Визуализация результатов
plt.plot(np.arange(train_size, timesteps), test_data, label='Исходные данные', marker='o')
plt.plot(np.arange(train_size + n_steps, timesteps), y_pred, label='Прогноз', marker='x')
plt.legend()
plt.show()