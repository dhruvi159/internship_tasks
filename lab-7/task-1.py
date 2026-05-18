from pprint import pprint

class StudentInfo:
    def __init__(self, rollno, name):
        self.student_rollno =  rollno
        self.student_name = name
    
    def display(self, students):
        pprint(students)
    

class StudentMarks:
    def __init__(self, rollno, marksone, markstwo, marksthree):
        self.student_rollno = rollno
        self.marks_one = marksone
        self.marks_two = markstwo
        self.marks_three = marksthree
        self.total_marks = marksone + markstwo + marksthree
        self.grade = self.calculate_grade()
        self.average - self.calculate_average()
        self.grade = self.calculate_grade()

    def calculate_average(self):
        return self.total_marks / 3
    
    def calculate_grade(self):
        if self.average >= 90:
            return "A"
        elif self.average >= 80:
            return "B"
        elif self.average >= 70:
            return "C"
        elif self.average >= 60:
            return "D"
        else:
            return "F"
    
    def display(self, marks):
        pprint(marks)
    

if __name__ == "__main__":

    students = []  
    marks = []     

    n = int(input("Enter number of students: "))

    for i in range(n):
        rollno = int(input("Enter student roll number: "))
        name = input("Enter student name: ")
    
        marksone = int(input("Enter data for student Marks of subject 1: "))
        markstwo = int(input("Enter data for student Marks of subject 2: "))
        marksthree = int(input("Enter data for student Marks of subject 3: "))

        obj1 = StudentInfo(rollno, name)
        obj2 = StudentMarks(rollno, marksone, markstwo, marksthree)

        students.append(obj1)
        marks.append(obj2)


    print("\n--- All Students Data ---")
    for s in students:
        s.display(s.__dict__)
    
    print("\n--- All Students Marks ---")
    for m in marks:
        m.display(m.__dict__)
    