import os
import shutil

data_dir = '/opt/ml/real_input/images'
profiles = os.listdir(data_dir)

before_age = ["58", "59"]
change_age = "60"


f = open("list_change_ages.txt", 'w')
for profile in profiles:
    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue
    id, gender, race, age = profile.split("_")
    if age in before_age:
        
        new_file_name = id + "_" + gender + "_" + race + "_" + change_age

        img_folder = os.path.join(data_dir, profile)
        new_img_folder = os.path.join(data_dir, new_file_name)

        store = id + "_" + gender + "_" + race + "_" + age + "_" + change_age + "\n"
        f.write(store)

        shutil.move(img_folder, new_img_folder)

f.close()

