#udf function

#1. Wordwise Reverse for the given string

def wordwise_reverse(s):
    reverse = s[::-1]
    print(reverse)

def interchange(s, pos1, pos2):
    l = list(s)
    l[pos1], l[pos2] = l[pos2], l[pos1]
    final_str = ''.join(l)
    print(final_str)

str = "This is the string example"

# wordwise_reverse(str)
# interchange(str, 0, 5)



def concatenate_lists(*lists):
    result = []
    for lst in lists:
        result.extend(lst)
    print("Concatenated List:", result)
    return result

def sum_of_lists(*lists):
    total_sum = 0
    for lst in lists:
        total_sum += sum(lst)
    print("Sum of all lists:", total_sum)
    return total_sum

def max_min_of_lists(*lists):
    all_elements = []
    for lst in lists:
        all_elements.extend(lst)
    print(f"Maximum: {max(all_elements)}, Minimum: {min(all_elements)}")
    return max(all_elements), min(all_elements)

# Lambda function for square of elements
square_of_lists = lambda *list: print("Squared List:", [item ** 2 for item in [x for lst in list for x in lst]])

# Lambda function for odd numbers
odd_numbers_of_lists = lambda *lists: print("Odd Numbers:", [num for lst in lists for num in lst if num % 2 != 0])


n = int(input("Enter number of lists: "))

all_lists = []

# Step 2: Take input for each list
for i in range(n):

    # User enters number of elements
    size = int(input(f"\nEnter number of elements for list {i+1}: "))

    temp = []

    # User enters elements
    print(f"Enter elements for list {i+1}:")

    for j in range(size):
        value = int(input())
        temp.append(value)

    # Store list inside all_lists
    all_lists.append(temp)

# Check if only one list was entered
if n == 1:
    print("List:", all_lists[0])
else:
    concatenate_lists(*all_lists)
    sum_of_lists(*all_lists)
    max_min_of_lists(*all_lists)
    square_of_lists(*all_lists)
    odd_numbers_of_lists(*all_lists)






