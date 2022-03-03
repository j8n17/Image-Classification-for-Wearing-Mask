import os

data_dir = '/opt/ml/real_input/images_sampling'
profiles = os.listdir(data_dir)

sec_one = 0
sec_one_M = 0
sec_one_F = 0

sec_two = 0
sec_two_M = 0
sec_two_F = 0

sec_three = 0
sec_three_M = 0
sec_three_F = 0

for profile in profiles:
    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue
    id, gender, race, age = profile.split("_")
    age = int(age)
    if age<30:
        sec_one += 1
        if gender == "female":
            sec_one_F += 1
        else:
            sec_one_M += 1

    elif age>=30 and age<60:
        sec_two += 1
        if gender == "female":
            sec_two_F += 1
        else:
            sec_two_M += 1

    elif age==60:
        sec_three += 1
        if gender == "female":
            sec_three_F += 1
        else:
            sec_three_M += 1

f = open("current_ages_in_sampling.txt", 'w')
f.write(f"less than 30 : {sec_one} \n")
f.write(f"\t female : {sec_one_F} \n")
f.write(f"\t male : {sec_one_M} \n\n")

f.write(f"more than 30 and less than 60 : {sec_two} \n")
f.write(f"\t female : {sec_two_F} \n")
f.write(f"\t male : {sec_two_M} \n\n")

f.write(f"60 years old : {sec_three} \n")
f.write(f"\t female : {sec_three_F} \n")
f.write(f"\t male : {sec_three_M} \n")

f.close()
