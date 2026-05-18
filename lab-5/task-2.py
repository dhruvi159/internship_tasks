
input_variables = input("Enter the variables (comma separated): ")

list_of_variables = input_variables.split(",")

int_list = []
str_list = []

for i in range(len(list_of_variables)):
    if list_of_variables[i].strip().isdigit():
        int_list.append(int(list_of_variables[i].strip()))
    elif list_of_variables[i].strip().isalpha():
        str_list.append(list_of_variables[i].strip())
    else:
        print(f"{list_of_variables[i].strip()} is neither an integer nor a string.")


print(f"min and max of the integer list: {min(int_list)}, {max(int_list)}")

reverse_str_list = str_list[::-1]

print(f"Revered string list: {reverse_str_list}")
