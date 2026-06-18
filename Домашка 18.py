# Классификация
# 1. Загрузка изображения
# 2. Масштабирование
# 3. Нормализация
# 4. Выбор модели
# 5. Загрузка изображения в модель и получение предсказания

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.application.resnet50 import preprocess_input
from tensorflow.keras.application.resnet50 import ResNet50
from tensorflow.keras.application.resnet50 import decode_prediction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.application.mobilnet import MobileNet
from tensorflow.keras.optimizers import Adam
import math



img_path = 'cat.png'
img = image.load_img(img_path, target_size=(224, 224))

img_array = image.img_to_array(img)
print(img_array[100, 100])
print(img_array.shape)
print(np.min(img_array))
print(np.max(img_array))

img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)
print(img_preprocessed.shape)
print(img_preprocessed[0, 100, 100])

model = ResNet50()
prediction = model.predict(img_preprocessed)
print(prediction)
print(decode_prediction(prediction))

# Название папок = название категорий
TRAIN_DATA_DIR = 'train_data'
VALIDATION_DATA_DIR = 'val_data'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500

# 'кошка или собака' -> 'кошка или НЕ кошка' - бинарная классификация
# 'кошка или собака' - мультиклассовая классификация

NUM_CLASSES = 2

IMG_WIDTH = 224
IMG_HEIGHT = 224

# Сколько изображений модель при обучении принимает одновременно
BATCH_SIZE = 64

# Аугментация - процедура увеличения кол-ва данных путём их "искажения": повороты, сдвиги, масштабирования
# Аугментация и нормализация
train_datagen = image. ImageDataGenerator (preprocessing_function=preprocess_input, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)

# только нормализация
val_datagen = image.ImageDataGenerator (preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(TRAIN_DATA_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, shuffle=True, seed=1, class_mode='categorical')
val_gen = train_datagen.flow_from_directory(VALIDATION_DATA_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, shuffle=False, seed=1, class_mode='categorical')

model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
for layer in model.layer[:]:
    layer.traible = False

input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

custom_model = model(input)
custom_model = GlobalAveragePooling2D()(custom_model)
custom_model = Dense(64, activation='relu')(custom_model)
custom_model = Dropout(0.5)(custom_model)
custom_model = Dense(NUM_CLASSES, activation='softmax')(custom_model)

target_model = Model(inputs=input, outputs=prediction)

target_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
num_steps =math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)
model.fit(
    train_gen,
    steps_per_epoch=num_steps,
    epochs=7,
    validation_data=val_gen,
    validation_steps=num_steps
)

target_model.save('our_model.h5')


