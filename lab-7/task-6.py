filename = input("Enter the filename to read: ")
file_location = input("Enter the file location: ")
input_lines = int(input("Enter the number of lines to read: "))
input_data = input("enter data to write in file: ")

with open(f"{file_location}/{filename}", "w") as file:
    file.write(input_data + "\n")   

try:
    filepath = f"{file_location}/{filename}"
    lines = []
    with open(filepath, "r") as file:
        for i in range(input_lines):
            line = file.readline()
            if not line:
                break
            lines.append(line.strip())
            print(line.strip())
    
    # Reverse the order of lines
    lines.reverse()
    
    # Write reversed lines to dummy.txt
    with open("dummy.txt", "w") as dummy_file:
        for line in lines:
            dummy_file.write(line + "\n")
    
    print(f"\nLines reversed and written to dummy.txt")
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found in '{file_location}'.")
except ValueError:
    print("Error: Number of lines must be an integer.")
