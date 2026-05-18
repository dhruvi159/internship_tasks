
student = {}

user = int(input("Enter number of students: "))

for i in range(user):
    student_roll = int(input("Enter student roll number: "))
    student_name = input("Enter student name: ")
    student_marks = int(input("Enter student marks: "))
    student[student_roll] = {"name": student_name, "marks": student_marks}

print(student)
