# Video Highlight Detection — MR.HiSum

Учебный ML-проект по автоматическому поиску хайлайтов в видео.

Задача формулируется как регрессия: для каждого временного шага видео модель предсказывает **highlight score** в диапазоне `[0, 1]`. В качестве целевой переменной используется `gtscore` из MR.HiSum.

## Идея эксперимента

Основная гипотеза: для поиска хайлайтов полезен временной контекст. Поэтому сравниваются две модели на одинаковых RGB-признаках:

- **MLP baseline** — оценивает каждый временной шаг независимо;
- **Temporal Transformer** — обрабатывает последовательность и учитывает отношения между временными шагами.

## Данные

- **MR.HiSum** — целевые `gtscore`;
- **YouTube-8M frame features** — предвычисленные RGB-признаки размерности 1024;
- в эксперименте используется подвыборка из первых **1000 видео**;
- разбиение выполняется **по видео**, а не по отдельным временным шагам: 70% train / 15% validation / 15% test;
- random seed: `42`.

Аудиопризнаки также извлекаются при подготовке данных, но в основном сравнительном эксперименте используются только RGB-признаки.

## Модели

### MLP baseline

```
1024 → 512 → ReLU → Dropout(0.3)
     → 128 → ReLU → Dropout(0.2)
     → 1 → Sigmoid
```

Loss: MSE. Optimizer: Adam, learning rate `1e-3`.

### Temporal Transformer

```
RGB features (1024)
→ Linear projection (128)
→ sinusoidal positional encoding
→ Transformer Encoder × 2
   - 4 attention heads
   - feed-forward dimension 256
→ Linear 128→64
→ GELU + Dropout
→ Linear 64→1
→ Sigmoid
```

Видео имеют разную длину, поэтому последовательности дополняются padding, а loss считается только по реальным временным шагам.

Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`. Используется gradient clipping.

## Метрики

- **MSE** — ошибка предсказания абсолютного значения highlight score;
- **Global Spearman correlation** — качество ранжирования временных шагов на объединённом test set;
- **Mean per-video Spearman correlation** — Spearman считается отдельно внутри каждого видео и затем усредняется.

Per-video Spearman особенно важен для задачи хайлайтов, поскольку на практике моменты ранжируются внутри конкретного видео.

## Результаты

| Model | Test MSE ↓ | Global Spearman ↑ | Mean per-video Spearman ↑ |
|---|---:|---:|---:|
| MLP | 0.08636 | 0.0254 | -0.0126 |
| Temporal Transformer | **0.07499** | **0.1037** | **0.0733** |

Temporal Transformer уменьшил test MSE примерно на **13.2%** относительно MLP. На per-video анализе Transformer показал более высокий Spearman на **82 из 150** тестовых видео и меньший MSE на **115 из 150** видео.

Результаты показывают пользу временного контекста, однако качество остаётся неоднородным между видео. Среди характерных ошибок наблюдаются сглаживание пиков, сжатие амплитуды предсказаний и неверное воспроизведение временного тренда.

## Структура репозитория

```
.
├── select_files.py          # определение необходимых YouTube-8M файлов
├── download_required.py     # загрузка необходимых TFRecord
├── extract_features.py      # извлечение RGB/audio features
├── train_mlp.py             # обучение MLP baseline
├── evaluate_mlp_video.py    # per-video оценка MLP
├── train_transformer.py     # обучение Temporal Transformer
├── analyze_results.py       # сравнение моделей и построение графиков
├── requirements.txt
├── .gitignore
└── README.md
```

## Запуск

Проект выполнялся с Python 3.11. Скрипты содержат локальную переменную `PROJECT_DIR`; перед запуском её необходимо заменить на путь к проекту на вашей машине.

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Последовательность запуска:

```bash
python select_files.py
python download_required.py
python extract_features.py
python train_mlp.py
python evaluate_mlp_video.py
python train_transformer.py
python analyze_results.py
```

Для подготовки признаков необходимы `metadata.csv`, файл разметки MR.HiSum и официальный YouTube-8M download plan. Большие датасеты, TFRecord, HDF5-файлы и обученные веса моделей намеренно не хранятся в Git.

## Воспроизводимость

Эксперимент использует фиксированный seed `42` и одинаковое video-level разбиение для MLP и Transformer. Лучшие checkpoints выбираются по validation loss.

Проект запускался локально на CPU под Windows с Python 3.11.

## Ограничения и дальнейшая работа

Текущий эксперимент ограничен 1000 видео и использует только RGB-признаки. Возможные направления развития: использование большего объёма MR.HiSum, объединение RGB и audio, ranking-aware loss, дополнительная регуляризация и подбор гиперпараметров Transformer.
