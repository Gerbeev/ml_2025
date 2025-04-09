import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_PATH = "model.keras"
EPOCH = 5  # Глобальное количество эпох для обучения/дообучения

def sanitize_name(name):
    """
    Заменяет все символы, кроме букв, цифр и '_', на '_'
    """
    return re.sub(r'[^0-9a-zA-Z_]', '_', name)

def load_csv_clean(file_path):
    """
    Загружает CSV, оставляет столбцы типов number, object, bool.
    Приводит object и bool к строке, а потом переименовывает столбцы через sanitize.
    """
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=["number", "object", "bool"])

    # Приводим все object/bool колонки к строковому представлению
    for col in df.select_dtypes(include=["object", "bool"]).columns:
        df[col] = df[col].astype(str).fillna("missing")
    # Для числовых колонок пытаемся привести к числу (если вдруг там строки)
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Удаляем строки с NaN в числовых колонках (в обучающем датасете таких быть не должно)
    df = df.dropna(subset=df.select_dtypes(include=["int64", "float64"]).columns)
    
    # Переименовываем столбцы в безопасные имена
    new_columns = {col: sanitize_name(col) for col in df.columns}
    df = df.rename(columns=new_columns)
    return df

def build_preprocessor_model(df):
    """
    Строит интегрированную модель-предобработчик.
    Для числовых столбцов – нормализация;
    для категориальных – StringLookup + CategoryEncoding.
    
    Возвращает:
      - inputs: словарь Keras Input-слоёв (ключи совпадают с именами из DataFrame)
      - concatenated: объединённый вектор признаков, который будет входом для автоэнкодера
    """
    inputs = {}
    processed_features = []
    
    # Числовые колонки
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numeric_cols:
        inp = layers.Input(shape=(1,), name=col)
        norm = layers.Normalization(name=f"{col}_norm")
        norm.adapt(df[col].values.reshape(-1, 1))
        processed = norm(inp)
        inputs[col] = inp
        processed_features.append(processed)
    
    # Категориальные колонки
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        inp = layers.Input(shape=(1,), name=col, dtype=tf.string)
        lookup = layers.StringLookup(name=f"{col}_lookup")
        lookup.adapt(df[col].values)
        encoded = lookup(inp)
        num_tokens = lookup.vocabulary_size()
        onehot = layers.CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot", name=f"{col}_onehot")
        processed = onehot(encoded)
        inputs[col] = inp
        processed_features.append(processed)
    
    # Если есть несколько признаков – объединяем их
    if len(processed_features) > 1:
        concatenated = layers.concatenate(processed_features, name="concatenated_features")
    else:
        concatenated = processed_features[0]
    
    return inputs, concatenated

def build_autoencoder_model(input_dim, encoding_dim=32):
    """
    Строит автоэнкодер, принимающий на вход вектор размерности input_dim.
    """
    ae_input = layers.Input(shape=(input_dim,), name="ae_input")
    encoded = layers.Dense(encoding_dim, activation="relu")(ae_input)
    decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)
    autoencoder = models.Model(inputs=ae_input, outputs=decoded, name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

def build_full_model(df, encoding_dim=32):
    """
    Строит полную модель с интегрированным препроцессингом и автоэнкодером.
    Вход – словарь с данными; на выходе – реконструированный вектор признаков.
    
    Для обучения целевым значением является результат встроенного препроцессинга.
    """
    inputs, preprocessed = build_preprocessor_model(df)
    input_dim = preprocessed.shape[-1]
    autoencoder = build_autoencoder_model(input_dim, encoding_dim)
    
    reconstruction = autoencoder(preprocessed)
    full_model = models.Model(inputs=inputs, outputs=reconstruction, name="full_autoencoder")
    full_model.compile(optimizer="adam", loss="mse")
    return full_model

def df_to_model_inputs(df, input_keys):
    """
    Преобразует DataFrame в словарь для подачи в модель.
    Для столбцов с типом object формируем тензор с dtype=tf.string,
    для остальных – тензор с dtype=tf.float32.
    """
    data = {}
    for key in input_keys:
        col = df[key]
        if col.dtype == "object":
            arr = tf.convert_to_tensor(col.astype(str).tolist(), dtype=tf.string)
            arr = tf.reshape(arr, (-1, 1))
        else:
            arr = tf.convert_to_tensor(col.to_numpy().astype(np.float32), dtype=tf.float32)
            arr = tf.reshape(arr, (-1, 1))
        data[key] = arr
    return data

def train_model_on_files(file_list, epochs=EPOCH):
    """
    Обучает модель на списке файлов.
    """
    all_dfs = [load_csv_clean(f) for f in file_list]
    df = pd.concat(all_dfs, ignore_index=True)
    
    full_model = build_full_model(df, encoding_dim=32)
    input_keys = list(full_model.input.keys())
    X = df_to_model_inputs(df, input_keys)
    
    # Получаем целевое значение через встроенную подсеть препроцессинга
    inputs, preprocessed = build_preprocessor_model(df)
    preprocessor_model = models.Model(inputs=inputs, outputs=preprocessed, name="preprocessor")
    y = preprocessor_model.predict(X)
    
    full_model.fit(X, y, epochs=epochs, batch_size=32, shuffle=True, verbose=1)
    full_model.save(MODEL_PATH)
    print(f"\n✅ Модель сохранена в {MODEL_PATH}")

def continue_training(file_list, epochs=EPOCH):
    """
    Дообучает модель на новых данных.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не найдена. Сначала обучите модель.")
    
    full_model = tf.keras.models.load_model(MODEL_PATH, compile=True)
    
    all_dfs = [load_csv_clean(f) for f in file_list]
    df = pd.concat(all_dfs, ignore_index=True)
    input_keys = list(full_model.input.keys())
    X = df_to_model_inputs(df, input_keys)
    
    # Получаем целевые признаки через уровень "concatenated_features"
    preprocessor_model = models.Model(inputs=full_model.input,
                                      outputs=full_model.get_layer("concatenated_features").output,
                                      name="preprocessor")
    y = preprocessor_model.predict(X)
    
    full_model.fit(X, y, epochs=epochs, batch_size=32, shuffle=True, verbose=1)
    full_model.save(MODEL_PATH)
    print(f"\n🔄 Дообучение завершено. Модель обновлена.")

def validate_file_with_autoencoder(file_path, threshold=0.01):
    """
    Валидирует новый файл на основе обученной модели.
    Строки с высокой ошибкой восстановления помечаются как невалидные.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не найдена.")
    
    full_model = tf.keras.models.load_model(MODEL_PATH, compile=True)
    df = load_csv_clean(file_path)
    
    input_keys = list(full_model.input.keys())
    X = df_to_model_inputs(df, input_keys)
    
    # Получаем целевые признаки через уровень "concatenated_features"
    preprocessor_model = models.Model(inputs=full_model.input,
                                      outputs=full_model.get_layer("concatenated_features").output,
                                      name="preprocessor")
    y_true = preprocessor_model.predict(X)
    
    # Ошибка восстановления
    y_pred = full_model.predict(X)
    loss = np.mean(np.square(y_true - y_pred), axis=1)
    
    # Помечаем строки с ошибкой восстановления выше порога
    df["ValidationLoss"] = loss
    df["IsValid"] = df["ValidationLoss"] < threshold
    
    print(df[["ValidationLoss", "IsValid"]].head())
    print(f"\n✅ Валидных записей: {(df['IsValid']).sum()} из {len(df)}")
    
    return df

# === ПРИМЕР ИСПОЛЬЗОВАНИЯ ===
if __name__ == "__main__":
    # Обучение модели на корректных данных:
    # train_model_on_files(["data1.csv", "data2.csv"])
    
    # Дообучение на новых данных:
    # continue_training(["data3.csv"])
    
    # Валидация нового файла (на вход могут подаваться любые данные):
    # validate_file_with_autoencoder("unseen_data.csv", threshold=0.02)
    pass
