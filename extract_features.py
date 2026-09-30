from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py


PROJECT_DIR = Path(r"C:\Users\dinat\mrhisum_project")

METADATA_PATH = PROJECT_DIR / "metadata.csv"
YT8M_DIR = PROJECT_DIR / "yt8m"

OUTPUT_PATH = PROJECT_DIR / "yt8m_features_1000.h5"

N_VIDEOS = 1000


# ============================================================
# 1. METADATA
# ============================================================

metadata = pd.read_csv(METADATA_PATH).iloc[:N_VIDEOS].copy()

metadata["random_id"] = metadata["random_id"].astype(str)
metadata["video_id"] = metadata["video_id"].astype(str)
metadata["yt8m_file"] = metadata["yt8m_file"].astype(str)

# YouTube-8M ID -> MR.HiSum video_N
random_to_video = dict(
    zip(
        metadata["random_id"],
        metadata["video_id"]
    )
)

wanted_ids = set(random_to_video)

print("=" * 60)
print("MR.HiSum FEATURE EXTRACTION")
print("=" * 60)

print(f"Видео в выборке: {len(metadata)}")
print(f"Уникальных random_id: {len(wanted_ids)}")


# ============================================================
# 2. TFRecord files
# ============================================================

files = sorted(YT8M_DIR.glob("*.tfrecord"))

print(f"TFRecord файлов: {len(files)}")

if not files:
    raise RuntimeError(
        f"TFRecord не найдены в {YT8M_DIR}"
    )


# ============================================================
# 3. Quantized YouTube-8M feature decoder
#
# YouTube-8M frame features хранятся как uint8.
# Восстанавливаем float features.
# ============================================================

def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):

    assert feat_vector.dtype == np.uint8

    quantized_range = max_quantized_value - min_quantized_value

    scalar = quantized_range / 255.0

    bias = (
        (quantized_range / 512.0)
        + min_quantized_value
    )

    return feat_vector.astype(np.float32) * scalar + bias


# ============================================================
# 4. Чтение SequenceExample
# ============================================================

def parse_record(raw_record):

    example = tf.train.SequenceExample()

    example.ParseFromString(
        raw_record.numpy()
    )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    context = example.context.feature

    video_id = (
        context["id"]
        .bytes_list
        .value[0]
        .decode("utf-8")
    )

    # --------------------------------------------------------
    # RGB frame features
    # --------------------------------------------------------

    feature_lists = example.feature_lists.feature_list

    rgb_frames = []

    if "rgb" in feature_lists:

        for feature in feature_lists["rgb"].feature:

            raw = feature.bytes_list.value[0]

            vector = np.frombuffer(
                raw,
                dtype=np.uint8
            )

            vector = dequantize(vector)

            rgb_frames.append(vector)

    if rgb_frames:

        rgb = np.stack(
            rgb_frames,
            axis=0
        )

    else:

        rgb = None

    # --------------------------------------------------------
    # AUDIO — сохраним тоже, пригодится для эксперимента
    # --------------------------------------------------------

    audio_frames = []

    if "audio" in feature_lists:

        for feature in feature_lists["audio"].feature:

            raw = feature.bytes_list.value[0]

            vector = np.frombuffer(
                raw,
                dtype=np.uint8
            )

            vector = dequantize(vector)

            audio_frames.append(vector)

    if audio_frames:

        audio = np.stack(
            audio_frames,
            axis=0
        )

    else:

        audio = None

    return video_id, rgb, audio


# ============================================================
# 5. Извлечение
# ============================================================

found = set()

with h5py.File(
    OUTPUT_PATH,
    "w"
) as output:

    for file_idx, tfrecord_path in enumerate(
        files,
        start=1
    ):

        print(
            f"\n[{file_idx}/{len(files)}] "
            f"{tfrecord_path.name}"
        )

        dataset = tf.data.TFRecordDataset(
            str(tfrecord_path)
        )

        file_found = 0

        for raw_record in dataset:

            random_id, rgb, audio = parse_record(
                raw_record
            )

            # Нас интересуют только MR.HiSum IDs
            if random_id not in wanted_ids:
                continue

            video_id = random_to_video[random_id]

            # Защита от дублей
            if video_id in output:
                continue

            group = output.create_group(
                video_id
            )

            group.attrs["random_id"] = random_id

            # --------------------------------------------
            # RGB
            # --------------------------------------------

            if rgb is not None:

                group.create_dataset(
                    "rgb",
                    data=rgb,
                    compression="gzip"
                )

            # --------------------------------------------
            # AUDIO
            # --------------------------------------------

            if audio is not None:

                group.create_dataset(
                    "audio",
                    data=audio,
                    compression="gzip"
                )

            found.add(random_id)

            file_found += 1

        print(
            f"  найдено нужных видео: {file_found}"
        )


# ============================================================
# 6. Результат
# ============================================================

missing = wanted_ids - found

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТ")
print("=" * 60)

print(f"Нужно было: {len(wanted_ids)}")
print(f"Найдено:    {len(found)}")
print(f"Не найдено: {len(missing)}")

print(f"\nФайл создан:")
print(OUTPUT_PATH)


if missing:

    print("\nПервые 20 отсутствующих random_id:")

    print(
        list(sorted(missing))[:20]
    )


# ============================================================
# 7. Проверяем созданный H5
# ============================================================

with h5py.File(
    OUTPUT_PATH,
    "r"
) as f:

    videos = list(f.keys())

    print("\nПример структуры:")

    for video_id in videos[:5]:

        print(f"\n{video_id}")

        group = f[video_id]

        for key in group.keys():

            print(
                f"  {key}: "
                f"{group[key].shape}"
            )