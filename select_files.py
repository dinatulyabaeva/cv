import pandas as pd
from pathlib import Path

METADATA = Path(r"C:\Users\dinat\mrhisum_project\metadata.csv")

# Пока берём 1000 видео
N_VIDEOS = 1000

df = pd.read_csv(METADATA)

print("Столбцы:")
print(df.columns.tolist())

print("\nВсего видео:", len(df))

sample = df.iloc[:N_VIDEOS].copy()

files = (
    sample["yt8m_file"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
)

print(f"\nБерём видео: {len(sample)}")
print(f"Для них требуется TFRecord-файлов: {len(files)}")

print("\nПервые файлы:")
print(files.head(20).tolist())

files.to_csv(
    r"C:\Users\dinat\mrhisum_project\required_files.txt",
    index=False,
    header=False
)

print("\nСписок сохранён в required_files.txt")