import os
import shutil
import random

# === ĐƯỜNG DẪN DATASET GỐC TỪ KAGGLE ===
KAGGLE_TRAIN = r"C:\XLAS\DoAn\DataSet_Kaggle\train"
KAGGLE_TEST  = r"C:\XLAS\DoAn\DataSet_Kaggle\test"

# === THƯ MỤC OUTPUT ===
OUTPUT_BASE = r"C:\XLAS\DoAn\XLAS_Project\dataset_new2"

TRAIN_OUT = os.path.join(OUTPUT_BASE, "train")
VALID_OUT = os.path.join(OUTPUT_BASE, "valid")
TEST_OUT  = os.path.join(OUTPUT_BASE, "test")

# === TỈ LỆ CHIA ===
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO  = 0.2


def ensure_dirs():
    os.makedirs(TRAIN_OUT, exist_ok=True)
    os.makedirs(VALID_OUT, exist_ok=True)
    os.makedirs(TEST_OUT, exist_ok=True)


def split_class(class_name, all_images):
    """Chia ảnh class theo 60/20/20"""

    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * TRAIN_RATIO)
    valid_end = int(total * (TRAIN_RATIO + VALID_RATIO))

    train_imgs = all_images[:train_end]
    valid_imgs = all_images[train_end:valid_end]
    test_imgs = all_images[valid_end:]

    # Tạo thư mục class
    os.makedirs(os.path.join(TRAIN_OUT, class_name), exist_ok=True)
    os.makedirs(os.path.join(VALID_OUT, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_OUT, class_name), exist_ok=True)

    # Copy
    for img in train_imgs:
        shutil.copy(img, os.path.join(TRAIN_OUT, class_name))

    for img in valid_imgs:
        shutil.copy(img, os.path.join(VALID_OUT, class_name))

    for img in test_imgs:
        shutil.copy(img, os.path.join(TEST_OUT, class_name))

    return len(train_imgs), len(valid_imgs), len(test_imgs)


def main():
    print("🚀 BẮT ĐẦU CHIA LẠI DATASET GỐC KAGGLE...")

    ensure_dirs()

    for class_name in os.listdir(KAGGLE_TRAIN):
        src_train = os.path.join(KAGGLE_TRAIN, class_name)
        src_test  = os.path.join(KAGGLE_TEST, class_name)

        if not os.path.isdir(src_train):
            continue

        print(f"\n📌 Xử lý class: {class_name}")

        # Gộp ảnh từ cả train + test
        all_imgs = []

        # Train set
        for f in os.listdir(src_train):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                all_imgs.append(os.path.join(src_train, f))

        # Test set
        for f in os.listdir(src_test):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                all_imgs.append(os.path.join(src_test, f))

        train_count, valid_count, test_count = split_class(class_name, all_imgs)

        print(f"  → Tổng ảnh: {len(all_imgs)}")
        print(f"    - Train: {train_count}")
        print(f"    - Valid: {valid_count}")
        print(f"    - Test : {test_count}")

    print("\n🎉 HOÀN THÀNH!")
    print("Dataset mới nằm tại:", OUTPUT_BASE)


if __name__ == "__main__":
    main()