#nested dictonary input

student = {}

n  = int(input("Enter number of students: "))

for i in range(n):
    roll_no = int(input("Enter student roll number: "))
    name = input("Enter student name: ")
    marks = int(input("Enter student marks: "))
    grade = None
    student[f"S{i+1}"] = {"roll_no": roll_no, "name": name, "marks": marks, "grade": grade}
    print("\n")
    if marks >=90 and marks <=100:
        student[f"S{i+1}"]["grade"] = "A"
    elif marks >=80 and marks<=90:
        student[f"S{i+1}"]["grade"] = "B"
    elif marks >=60 and marks <=80:
        student[f"S{i+1}"]["grade"] = "C"
    elif marks >=40 and marks <=60:
        student[f"S{i+1}"]["grade"] = "D"
    else:
        student[f"S{i+1}"]["grade"] = "FAIL"


import json
print(json.dumps(student, indent=4))