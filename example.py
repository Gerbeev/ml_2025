import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Embedding
from tensorflow.keras.models import Model

# ===== ШАГ 1. ЗАГРУЗКА CSV =====
df = pd.read_csv("transactions.csv")

# ===== ШАГ 2. Определяем типы столбцов =====
# Например, предположим, что числовые признаки имеют тип int/float, а категориальные – object
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Числовые признаки:", numerical_cols)
print("Категориальные признаки:", categorical_cols)

# ===== ШАГ 3. Создание препроцессинг слоёв для TensorFlow =====
# Для числовых признаков: Normalization
numerical_data = df[numerical_cols].values.astype("float32")
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numerical_data)

# Для категориальных признаков: StringLookup +, при необходимости, Embedding или OneHotEncoding.
# Здесь мы используем StringLookup для преобразования строки в целое число.
lookup_layers = {}
for col in categorical_cols:
    lookup = tf.keras.layers.StringLookup(output_mode='int')
    lookup.adapt(df[col])
    lookup_layers[col] = lookup

# ===== ШАГ 4. Построение модели с встроенным препроцессингом =====
# Вход для числовых признаков
input_numerical = Input(shape=(len(numerical_cols),), name="numerical_input")
normalized_numerical = normalizer(input_numerical)

# Входы для категориальных признаков
categorical_inputs = []
categorical_embeddings = []  # можно использовать Embedding или OneHotEncoding

for col in categorical_cols:
    inp = Input(shape=(1,), dtype=tf.string, name=f"{col}_input")
    # Преобразуем строку в целое число
    lookup = lookup_layers[col]
    idx = lookup(inp)
    # Пример: если количество категорий велико, можно использовать Embedding
    vocab_size = len(lookup.get_vocabulary())
    embed = Embedding(input_dim=vocab_size, output_dim=4, name=f"{col}_embedding")(idx)
    flat_embed = Flatten()(embed)
    categorical_inputs.append(inp)
    categorical_embeddings.append(flat_embed)

# Объединяем все признаки
if categorical_embeddings:
    concatenated = Concatenate(name="concat")([normalized_numerical] + categorical_embeddings)
else:
    concatenated = normalized_numerical

# Пример простой архитектуры автоэнкодера
encoded = Dense(8, activation="relu", name="encoder")(concatenated)
decoded = Dense(concatenated.shape[-1], activation="sigmoid", name="decoder")(encoded)

# Собираем модель
model = Model(inputs=[input_numerical] + categorical_inputs, outputs=decoded)
model.compile(optimizer="adam", loss="mse")
model.summary()

# ===== ШАГ 5. Подготовка данных для обучения =====
# Для числовые признаки – непосредственно массив
X_num = df[numerical_cols].values.astype("float32")

# Для категориальных – оставляем строки
X_cat = [df[col].values for col in categorical_cols]

# Целевая переменная для автоэнкодера – это же объединённые данные
# Здесь мы строим цель на основе числовых и категориальных данных
# Для упрощения, в данном примере целевая часть – только числовые данные, но можно расширить.
# Если нужно восстанавливать и категориальные признаки, можно использовать другой подход.
y = normalizer(X_num)  # или можно использовать np.hstack(...)

# ===== ШАГ 6. Обучение модели =====
model.fit(
    [X_num] + X_cat,
    y,
    epochs=50,
    batch_size=32,
    validation_split=0.1
)

# ===== ШАГ 7. Сохранение модели =====
model.save("my_autoencoder_model")

# Теперь модель полностью включает в себя этапы препроцессинга.
# Чтобы загрузить модель и применить её к новым данным, достаточно:
# loaded_model = tf.keras.models.load_model("my_autoencoder_model")




# Загрузка модели
loaded_model = tf.keras.models.load_model("my_autoencoder_model")

# Загрузка новых данных
new_df = pd.read_csv("new_transactions.csv")
X_new_num = new_df[numerical_cols].values.astype("float32")
X_new_cat = [new_df[col].values for col in categorical_cols]

# Получаем предсказание (например, для оценки ошибки реконструкции)
reconstructed = loaded_model.predict([X_new_num] + X_new_cat)

# Можно вычислить MSE или другие метрики для определения аномалий
mse = np.mean(np.power(normalizer(X_new_num) - reconstructed, 2), axis=1)
print(mse)
