class MaximumLimitError(Exception):
    '''Custom exception for exceeding maximum withdrawal limit.'''
    pass

class MaximumTransactionError(Exception):
    '''Custom exception for exceeding maximum transaction limit.'''
    pass


class HDFCBank:
    def __init__(self):
        self.__balance = 100000  
        self.__max_limit = 20000
        self.__max_transactions = 3
        self.__transaction_count = 0
    
    def withdraw(self, amount):
        # Check transaction limit
        if self.__transaction_count >= self.__max_transactions:
            raise MaximumTransactionError(f"HDFC Bank: Transaction limit exceeded ({self.__max_transactions} transactions).")
        
        
        if amount > self.__max_limit:
            raise MaximumLimitError(f"HDFC Bank: Withdrawal amount exceeds the maximum limit of {self.__max_limit} rupees.")
        

        if amount > self.__balance:
            raise MaximumLimitError(f"HDFC Bank: Insufficient balance. Available: {self.__balance} rupees.")
        
        # Deduct amount
        self.__balance -= amount
        self.__transaction_count += 1
        print(f"HDFC Bank: Successfully withdrawn {amount} rupees.")
        print(f"  Transactions used: {self.__transaction_count}/{self.__max_transactions}\n")
    
    def get_balance(self):
        return self.__balance
    
    def get_transaction_count(self):
        return self.__transaction_count


class AXISBank:
    def __init__(self):
        self.__balance = 150000 
        self.__max_limit = 30000
        self.__max_transactions = 5
        self.__transaction_count = 0
    
    def withdraw(self, amount):
        if self.__transaction_count >= self.__max_transactions:
            raise MaximumTransactionError(f"AXIS Bank: Transaction limit exceeded ({self.__max_transactions} transactions).")
        
    
        if amount > self.__max_limit:
            raise MaximumLimitError(f"AXIS Bank: Withdrawal amount exceeds the maximum limit of {self.__max_limit} rupees.")
        
      
        if amount > self.__balance:
            raise MaximumLimitError(f"AXIS Bank: Insufficient balance. Available: {self.__balance} rupees.")
        
       
        self.__balance -= amount
        self.__transaction_count += 1
        print(f" AXIS Bank: Successfully withdrawn {amount} rupees.")
        print(f"  Remaining Balance: {self.__balance} rupees")
        print(f"  Transactions used: {self.__transaction_count}/{self.__max_transactions}\n")
    
    def get_balance(self):
        return self.__balance
    
    def get_transaction_count(self):
        return self.__transaction_count


class ATM:
    def __init__(self):
        self.hdfc = HDFCBank()
        self.axis = AXISBank()
    
    def start(self):
        bank_choice = input("Choose the bank (HDFC/AXIS): ").upper()
        
        if bank_choice not in ["HDFC", "AXIS"]:
            print("Invalid bank choice.")
            return
        
        bank = self.hdfc if bank_choice == "HDFC" else self.axis
        
        while True:
            try:
                amount = int(input("Enter the amount to withdraw: "))
                
                if amount <= 0:
                    print("Please enter a valid amount.\n")
                    continue
                
                #each bank implements its own withdraw
                bank.withdraw(amount)
                
                # Ask for next transaction
                next_transaction = input("Do you want to perform another transaction? (yes/no): ").lower()
                if next_transaction != "yes":
                    print(f"Thank you for using {bank_choice} ATM. Goodbye!")
                    break
                print()
                
            except MaximumLimitError as e:
                print(f"Error: {e}\n")
                break
            except MaximumTransactionError as e:
                print(f"Error: {e}\n")
                break
            except ValueError:
                print("Please enter a valid number.\n")


if __name__ == "__main__":
    atm = ATM()
    atm.start()