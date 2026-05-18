#exception handling with choice menu

def pattern1():
    """Multiple try block with single except block"""
    print("\n1: Multiple Try Block with Single Except Block ---")
    try:
        print("First try block")
        n1 = int(input("Enter a number: "))
        print(f"You entered: {n1}")
    except Exception as e:
        print(f"Error in first try: {e}")

    try:
        print("Second try block")
        n2 = int(input("Enter another number: "))
        result = n1 / n2
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error in second try: {e}")


def pattern2():
    """Nested try with nested except block"""
    print("\n2: Nested Try with Nested Except Block ---")
    try:
        print("Outer try block")
        num1 = int(input("Enter first number: "))
        
        try:
            print("Inner try block")
            num2 = int(input("Enter second number: "))
            result = num1 / num2
            print(f"Division result: {result}")
        except ZeroDivisionError as e:
            print(f"Inner except: {e}")
        
    except ValueError as e:
        print(f"Outer except: {e}")


def pattern3():
    """Nested try with nested except with nested finally"""
    print("\n3: Nested Try with Nested Except with Nested Finally ---")
    try:
        print("Outer try block")
        x = int(input("Enter value for x: "))
        
        try:
            print("Inner try block")
            y = int(input("Enter value for y: "))
            calc = x / y
            print(f"Calculation: {calc}")
        except ZeroDivisionError as e:
            print(f"Inner except: Division by zero - {e}")
        finally:
            print("Inner finally block executed")
            
    except ValueError as e:
        print(f"Outer except: Invalid input - {e}")
    finally:
        print("Outer finally block executed")


def pattern4():
    """Multiple try block with multiple except block"""
    print("\n4: Multiple Try Block with Multiple Except Block ---")
    try:
        print("First try block")
        a = int(input("Enter number a: "))
        if a < 0:
            raise ValueError("Negative numbers not allowed")
        result1 = 10 / a
        print(f"First result: {result1}")
    except ValueError as e:
        print(f"First except (ValueError): {e}")
    except ZeroDivisionError as e:
        print(f"First except (ZeroDivisionError): {e}")

    try:
        print("\nSecond try block")
        b = int(input("Enter number b: "))
        result2 = 20 / b
        print(f"Second result: {result2}")
    except ZeroDivisionError as e:
        print(f"Second except (ZeroDivisionError): {e}")
    except Exception as e:
        print(f"Second except (General Exception): {e}")


def pattern5():
    """Multiple try with multiple except with multiple finally"""
    print("\n5: Multiple Try with Multiple Except with Multiple Finally ---")
    try:
        print("First try block")
        p = int(input("Enter number p: "))
        if p == 0:
            raise ValueError("Zero not allowed")
        r1 = 100 / p
        print(f"First calculation: {r1}")
    except ValueError as e:
        print(f"First except (ValueError): {e}")
    except ZeroDivisionError as e:
        print(f"First except (ZeroDivisionError): {e}")
    finally:
        print("First finally block executed\n")

    try:
        print("Second try block")
        q = int(input("Enter number q: "))
        r2 = 50 / q
        print(f"Second calculation: {r2}")
    except ZeroDivisionError as e:
        print(f"Second except (ZeroDivisionError): {e}")
    except ValueError as e:
        print(f"Second except (ValueError): {e}")
    finally:
        print("Second finally block executed\n")



while True:
    print("\n" + "="*60)
    print("EXCEPTION HANDLING PATTERNS - CHOOSE ONE")
    print("="*60)
    print("1. Multiple Try Block with Single Except Block")
    print("2. Nested Try with Nested Except Block")
    print("3. Nested Try with Nested Except with Nested Finally")
    print("4. Multiple Try Block with Multiple Except Block")
    print("5. Multiple Try with Multiple Except with Multiple Finally")
    print("6. Exit")
    print("="*60)
    
    choice = input("Enter your choice (1-6): ").strip()
    
    if choice == "1":
        pattern1()
    elif choice == "2":
        pattern2()
    elif choice == "3":
        pattern3()
    elif choice == "4":
        pattern4()
    elif choice == "5":
        pattern5()
    elif choice == "6":
        print("\nThank you for using Exception Handling Patterns. Goodbye!")
        break
    else:
        print("Invalid choice! Please enter a number between 1 and 6.")
    
    # Ask if user wants to continue
    cont = input("\nDo you want to try another pattern? (yes/no): ").strip().lower()
    if cont != "yes":
        print("\nThank you for using Exception Handling Patterns. Goodbye!")
        break
    

