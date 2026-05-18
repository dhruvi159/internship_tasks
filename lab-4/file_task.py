
def write_data(num_students):
    students_data = []

    for i in range(num_students):
        dict = {}
        dict['name'] = input(f"\nEnter name for {i+1}: \n")   
        dict['rollno'] = input(f"\nEnter rollno: \n") 
        

        dict['marks'] = []
        print(f"\nEnter marks for 3 subjects for student {i+1}: ")
        for j in range(3):
            dict['marks'].append(int(input(f"Enter marks for subject {j+1}: ")))

        # Calculate average
        average_marks = sum(dict['marks']) / len(dict['marks'])
        dict['average'] = average_marks

        # Determine grade based on average
        if average_marks >= 80:
            dict['grade'] = 'A'
        elif average_marks >= 60:
            dict['grade'] = 'B'
        elif average_marks >= 40:
            dict['grade'] = 'C'
        else:
            dict['grade'] = 'Fail'

        students_data.append(dict)

        with open("studentInfo.txt", "a") as file:
            file.write(f"{dict['name']}, {dict['rollno']}\n")

    # Sort students by average in descending order
    students_data.sort(key=lambda x: x['average'], reverse=True)

    # Write sorted data to studentMarks.txt
    with open("studentMarks.txt", "a") as file:
        for student in students_data:
            file.write(f"Roll No: {student['rollno']}, Name: {student['name']}, Average: {student['average']:.2f}, Grade: {student['grade']}\n")

            

num_students = int(input("Enter number of students: "))
write_data(num_students)