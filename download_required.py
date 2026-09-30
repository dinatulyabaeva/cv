from pathlib import Path
import json
import itertools
import string
import urllib.request
import time


PROJECT_DIR = Path(r"C:\Users\dinat\mrhisum_project")
FILES_LIST = PROJECT_DIR / "required_files.txt"
OUTPUT_DIR = PROJECT_DIR / "yt8m"

PLAN_PATH = Path(
    r"C:\Users\dinat\yt8m_train_plan\2_frame_train_download_plan.json"
)

MIRROR = "us"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Какие trainXXXX нужны MR.HiSum
# ============================================================

with open(FILES_LIST, "r", encoding="utf-8") as f:
    required = {
        line.strip()
        for line in f
        if line.strip()
    }

print(f"MR.HiSum требует файлов: {len(required)}")


# ============================================================
# 2. Читаем официальный download plan
# ============================================================

with open(PLAN_PATH, "r", encoding="utf-8") as f:
    plan = json.load(f)

physical_files = list(plan["files"].keys())

print(
    f"Файлов в YouTube-8M train plan: "
    f"{len(physical_files)}"
)


# ============================================================
# 3. Воспроизводим нумерацию официального download.py
# ============================================================

VOCAB = (
    list(string.ascii_lowercase)
    + list(string.ascii_uppercase)
    + list(string.digits)
)

file_ids = [
    "".join(x)
    for x in itertools.product(VOCAB, repeat=2)
]

file_index = {
    code: i
    for i, code in enumerate(file_ids)
}


mapping = {}

for physical_name in physical_files:

    fname, ext = physical_name.split(".")

    code = fname[-2:]

    if code not in file_index:
        continue

    normalized = (
        f"{fname[:-2]}"
        f"{file_index[code]:04d}"
    )

    mapping[normalized] = physical_name


# ============================================================
# 4. Проверяем соответствие
# ============================================================

available = sorted(
    x for x in required
    if x in mapping
)

missing = sorted(
    x for x in required
    if x not in mapping
)


print("\n" + "=" * 60)
print("ПРОВЕРКА MAPPING")
print("=" * 60)

print(
    f"Найдено: {len(available)} / {len(required)}"
)

print("\nПервые соответствия:")

for name in available[:20]:

    print(
        f"{name} -> {mapping[name]}"
    )


if missing:

    print("\nНЕ НАЙДЕНЫ:")

    for name in missing:
        print(name)

    raise RuntimeError(
        "Не все необходимые TFRecord найдены."
    )


# ============================================================
# 5. Загрузка
# ============================================================

def download(url, destination, retries=3):

    for attempt in range(1, retries + 1):

        try:

            print(f"URL: {url}")

            urllib.request.urlretrieve(
                url,
                destination
            )

            size_mb = (
                destination.stat().st_size
                / 1024
                / 1024
            )

            print(
                f"OK: {destination.name} "
                f"({size_mb:.2f} MB)"
            )

            return True

        except Exception as e:

            print(
                f"Ошибка {attempt}/{retries}: {e}"
            )

            if destination.exists():
                destination.unlink()

            time.sleep(3)

    return False


# ============================================================
# 6. Только необходимые MR.HiSum TFRecord
# ============================================================

success = 0
failed = []


for i, normalized in enumerate(
    available,
    start=1
):

    physical_name = mapping[normalized]

    destination = (
        OUTPUT_DIR
        / f"{normalized}.tfrecord"
    )

    print("\n" + "=" * 60)

    print(
        f"[{i}/{len(available)}] "
        f"{normalized}"
    )

    print(
        f"Физический файл: "
        f"{physical_name}"
    )


    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):

        size_mb = (
            destination.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"Уже скачан "
            f"({size_mb:.2f} MB)"
        )

        success += 1
        continue


    url = (
        f"http://{MIRROR}.data.yt8m.org/"
        f"2/frame/train/"
        f"{physical_name}"
    )


    if download(
        url,
        destination
    ):

        success += 1

    else:

        failed.append(normalized)


# ============================================================
# 7. Итог
# ============================================================

print("\n" + "=" * 60)
print("ЗАГРУЗКА ЗАВЕРШЕНА")
print("=" * 60)

print(f"Успешно: {success}")
print(f"Ошибок: {len(failed)}")

if failed:

    print("\nНе удалось скачать:")

    for name in failed:
        print(name)