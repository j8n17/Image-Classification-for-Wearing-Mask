import os
import random
import shutil

data_dir = '/opt/ml/input/data/train/images'
profiles = os.listdir(data_dir)
for profile in profiles:                # 이미지 폴더들 (사람별)
    if profile.startswith("."):         # "." 로 시작하는 파일은 무시합니다
        continue

    old_path = os.path.join(data_dir, profile)

    id, gender, race, age = profile.split("_")

    if id > '100000':
        break
    if int(age) >= 58 and int(age) < 60:
        new_name = '_'.join([id, gender, race, '60'])
        new_path = os.path.join(data_dir, new_name)
        shutil.move(old_path, new_path)
    elif int(age) >=  60:
        id = str(random.choice(list(range(100000, 999999))))
        new_name = '_'.join([id, gender, race, '60'])
        new_path = os.path.join(data_dir, new_name)
        if os.path.exists(new_path):
            id = str(random.choice(list(range(100000, 999999))))
            new_name = '_'.join([id, gender, race, '60'])
            new_path = os.path.join(data_dir, new_name)
        shutil.copytree(old_path, new_path)