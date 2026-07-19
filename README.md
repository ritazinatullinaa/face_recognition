# Face Recognition Pipeline

Проект посвящен построению полного пайплайна распознавания лиц на датасете CelebA: от подготовки и выравнивания лиц до обучения моделей распознавания и проверки качества embeddings.

## Что сделано

- Подготовлены данные CelebA: bbox, landmarks, identity и изображения лиц.
- Реализована модель для предсказания facial landmarks на основе Stacked Hourglass.
- Реализовано выравнивание лиц по найденным landmarks.
- Обучена модель распознавания лиц на EfficientNet-B0 с CrossEntropy loss.
- Реализован и обучен ArcFace-подход для получения более геометрически осмысленных embeddings.
- Собран полный pipeline: детекция лица, предсказание landmarks, alignment, получение embedding.
- Дополнительно реализован эксперимент с Triplet Loss.
- Качество embeddings сравнивается через cosine similarity, gap между same/different pairs и verification accuracy.

## Структура файлов

```text
.
├── 1_Face_Alignment.ipynb
├── 2_My_ArcFace.ipynb
├── 3_Full_Pipeline-6.ipynb
├── 5_Triplet_Loss-2.ipynb
├── aligned_faces_predicted.csv
├── processed_10000_faces.csv
├── stacked_hourglass_best.pt
├── ce_efficientnet_b0_best-2.pt
└── arcface_efficientnet_b0_best.pt
```

## Описание

### `1_Face_Alignment.ipynb`

Подготовка датасета и обучение модели для facial landmarks.

- загружаются таблицы CelebA;
- выбираются изображения и identity;
- выполняется crop лиц;
- landmarks переводятся в координаты кропа;
- строятся heatmap-targets;
- обучается Stacked Hourglass;
- лица выравниваются по ground truth и predicted landmarks.

Результаты:

- `processed_10000_faces.csv`;
- `aligned_faces_predicted.csv`;
- `stacked_hourglass_best.pt`.

### `2_My_ArcFace.ipynb`

Обучение моделей распознавания лиц.

- загружаются выровненные лица;
- обучается EfficientNet-B0 с CrossEntropy;
- реализуется ArcFace layer;
- обучается EfficientNet-B0 с ArcFace;
- сравнивается качество моделей;
- embeddings оцениваются через cosine similarity.

Результаты:

- `ce_efficientnet_b0_best-2.pt`;
- `arcface_efficientnet_b0_best.pt`.

### `3_Full_Pipeline-6.ipynb`

Полный inference pipeline для новых изображений:

1. Находит лицо с помощью MTCNN.
2. Делает crop лица.
3. Предсказывает landmarks через Stacked Hourglass.
4. Выравнивает лицо по landmarks.
5. Получает embedding через EfficientNet-B0.
6. Визуализирует bbox, landmarks и aligned face.

### `5_Triplet_Loss-2.ipynb`

Дополнительный эксперимент с Triplet Loss.

- реализован `TripletFaceDataset`;
- модель обучается на тройках `anchor`, `positive`, `negative`;
- используется cosine distance;
- качество оценивается не через classification accuracy, а через triplet accuracy и качество embeddings.

Результат:

- `triplet_efficientnet_b0_best.pt`

## Данные и веса

### `processed_10000_faces.csv`

Таблица с подготовленными cropped faces и координатами landmarks после crop/resize.

### `aligned_faces_predicted.csv`

Таблица с путями к лицам, выровненным по predicted landmarks.

### `stacked_hourglass_best.pt`

Лучшие веса модели Stacked Hourglass для предсказания facial landmarks.

### `ce_efficientnet_b0_best-2.pt`

Лучшие веса EfficientNet-B0, обученной с CrossEntropy loss.

### `arcface_efficientnet_b0_best.pt`

Лучшие веса EfficientNet-B0, обученной с ArcFace loss.

### `triplet_efficiientnet_b0_best.pt`

Лучшие веса EfficientNet-B0, обученной с Triplet loss.
