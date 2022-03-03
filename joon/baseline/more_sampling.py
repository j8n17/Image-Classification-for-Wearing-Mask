import os
import shutil

data_dir = '/opt/ml/real_input/images_sampling'
profiles = os.listdir(data_dir)

for profile in profiles:
    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue
    id, gender, race, age = profile.split("_")
    age = int(age)
    if age == 60:
        profile_path = os.path.join(data_dir, profile)
        new_profile = id + "-n" + "_" + gender + "_" + race + "_" + str(age)
        new_path = os.path.join(data_dir, new_profile)
        shutil.copytree(profile_path, new_path)
