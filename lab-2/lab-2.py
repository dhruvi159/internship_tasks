#string operations

str = 'this is the string example'

#1. reverse the string

print(str[::-1]) #using slicing to reverse the string

#2. word wise reverse the string

split = str.split() #splitting the string into words
reverse = split[::-1] #reversing the list of words
print(' '.join(reverse)) #joining the reversed list of words into a string 

#3. characters interchange

list = list(str) #converting the string into a list of characters

pos = 0 
pos2 = 5 

list[pos], list[pos2] = list[pos2], list[pos] #interchanging the characters at the specified positions
final_str = ''.join(list) #joining the list of characters back into a string
print(final_str) 


#4. space splitjoin the string with *
split2 = str.split() #splitting the string into words
join = '*'.join(split2) #joining the list of words into a string

print(join) #printing the string with "*"

#5. replace the word 'is' with 'was'
string3 = str.replace('is', 'was') #replacing the word 'is' with 'was' in the string and printing it

print(string3)


#replace only the word 'is' with 'was' and not the 'is' in 'this'

string = 'this is the string example' #defining the original string

split3 = string.split() #splitting the string into words

result = [] #creating an empty list to store the modified words

for substring in split3: #iterating through the list of words
    if substring == 'is':
        result.append('was') # append 'was' to the result list
    else:
        result.append(substring) #append the original word to the result list

final = ' '.join(result)

print(final)
