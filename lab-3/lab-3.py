#single line loop

# ls = ["hello" for i in range(5)]
# print(ls)

# nested loop

# for i in range(10):
#     for j in range(20, 30):
#         print("j >>>>>>>>>>>>>>>", j)
#     print("i", i)

#1. user input a character

char = input("Enter a character: ") 

if char.isalpha(): 
    print("The character is an alphabet", char)
else:
    print("Invalid character")


#2 user input a number 

num1 = int(input("Enter a number: "))
num2 = int(input("Enter another number: "))

if num1 > 0 and num2 > 0:
    print("Both numbers are positive")
    if num1 % 2 != 0 and num2 % 2 != 0:
        print("sum of the odd numbers is", int(num1) + int(num2))
    elif num1 % 2 == 0 and num2 % 2 == 0:
        print("multiplication of the even numbers is", float(num1) * float(num2))
    else:
        print("One number is odd and the other is even")
elif num1 < 0 and num2 < 0:
    print("Both numbers are negative")
    if num1 % 2 != 0 and num2 % 2 != 0:
        print("Substraction of the odd numbers is", int(num1) - int(num2))
    elif num1 % 2 == 0 and num2 % 2 == 0:
        print("division of the even numbers is", float(num1) // float(num2))
    else:
        print("One number is odd and the other is even")
else:
    print("One number is positive and the other is negative")


#user input marks

marks = int(input("Enter your marks: "))

if marks >= 90 and marks <= 100:
    print("Grade A")
elif marks >= 80 and marks < 90:
    print("Grade B")
elif marks >= 60 and marks < 80:
    print("Grade C")
elif marks >= 40 and marks < 60:
    print("Grade D")
elif marks < 40:
    print("FAIL")
else:
    print("Invalid marks")

#patterns 

# printing pattern of 1's 
n = int(input("Enter the number of rows: "))

for i in range(1, n+1):
    for j in range(1, i+1):
        print("1", end="")
    print()

#printing pattern of chronology of numbers 

m = 1
for i in range(1,n+1):
     for j in range(1, i+1):
         print(m, end=" ")
         m += 1
     print()

#printing inverse star pattern

for i in range(n, 0, -1):
    for j in range(1, i+1):
        print("*", end=" ")
    print()

#triangle pyramid pattern

k = (2*n) - 2 #number of spaces
for i in range(0, n):
    for j in range(0, k):
        print(end=" ")
        
    k = k - 1
    for j in range(0, i+1):
        print("* ", end=" ")
    print(" ")


#Diamond shaped pattern

d = (2*n) - 2 #number of spaces
for i in range(0, n):
    for j in range(0, d):
        print(end=" ")
    
    d = d - 1
    for j in range(0, i + 1):
        print("* ", end="")
    print("")
    
d = n - 2

for i in range(n, -1, -1):
    for j in range(d, 0, -1):
        print(end=" ")

    d = d + 1
    for j in range(0, i+1):
        print("* ", end="")
    print("")


#factorial program

num = int(input("Enter a number: "))

fact = 1

while num > 0:
    fact *= num
    num -= 1
print("Factorial is", fact)
