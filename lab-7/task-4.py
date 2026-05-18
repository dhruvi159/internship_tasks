#inheritance 

import random

class BankInfo:
    def __init__(self, fn, ln, gender, address):
        self.fn = fn
        self.ln = ln
        self.gender = gender
        self.address = address
    
class BankAccount(BankInfo):
    def __init__(self, fn, ln, gender, address, acno, amount):
        super().__init__(fn, ln, gender, address)
        self.acno = acno 
        self.amount = amount
    
class saving(BankAccount):
    min_amount = 10000
    rate = 0.06
    def __init__(self, fn, ln, gender, address, acno, amount):
        super().__init__(fn, ln, gender, address, acno, amount)
    
    def calculate_interest(self, months):
        """Calculate amount with interest based on months"""
        monthly_rate = self.rate / 12
        interest_amount = self.amount * monthly_rate * months
        total = self.amount + interest_amount
        return interest_amount, total 
    
    def display(self):
        print("\n----- Saving Account Details -----")
        print(f"First Name: {self.fn}")
        print(f"Last Name: {self.ln}")
        print(f"Gender: {self.gender}")
        print(f"Address: {self.address}")
        print(f"Account Number: {self.acno}")
        print(f"Amount: {self.amount}")

class current(BankAccount):
    min_amount = 5000
    rate = None
    def __init__(self, fn, ln, gender, address, acno, amount):
        super().__init__(fn, ln, gender, address, acno, amount)
    
    def display(self):
        print(f"First Name: {self.fn}")
        print(f"Last Name: {self.ln}")
        print(f"Gender: {self.gender}")
        print(f"Address: {self.address}")
        print(f"Account Number: {self.acno}")
        print(f"Amount: {self.amount}")
    
if __name__ == "__main__":

    fn = input("Enter first name: ")
    ln = input("Enter last name: ")
    gender = input("Enter gender: ")
    address = input("Enter address: ")
    acno = random.randint(1000000000000, 9999999999999)
    

    choice = input(print("Select saving account or current account")).lower()

    chances = 3

while chances > 0:
    amount = float(input("Enter initial amount: "))
    
    if choice == "saving":
        print("\n--- Saving Account ---")
        if amount >= saving.min_amount:
            saving_account = saving(fn, ln, gender, address, acno, amount)
            saving_account.display()

            months = int(input("Enter number of months:  "))
            interest, total = saving_account.calculate_interest(months)
            print(f"Total amount after {months} months: {total:.2f}")

            break
        else:
            chances -= 1
            print(f"Amount must be at least {saving.min_amount}. You have {chances} chances left.\n")
    elif choice == "current":
        print("\n--- Current Account ---")
        if amount >= current.min_amount:
            current_account = current(fn, ln, gender, address, acno, amount)
            current_account.display()
            break
        else:
            chances -= 1
            print(f"Amount must be at least {current.min_amount}. You have {chances} chances left.\n")
    else:
        print("Invalid account type. Please choose 'saving' or 'current'.")
        break

    if chances == 0:    
        print("You have exhausted all chances. Please try again later.")