class passwordError(Exception):
    '''Custom exception for password validation errors.'''
    pass



class SignUp:
    def __init__(self, fn, ln, username, password):
        self.fn = fn
        self.ln = ln
        self.username = username
        self.password = self.validate_password(password)

    def validate_password(self, password):
        """Validate password based on criteria"""
        if len(password) < 8 or len(password) > 16:
            raise passwordError("Password must be between 8 and 16 characters long.")
        if not any(char.isupper() for char in password):
            raise passwordError("Password must contain at least one uppercase letter.")
        if not any(char.islower() for char in password):
            raise passwordError("Password must contain at least one lowercase letter.")
        if not any(char.isdigit() for char in password):
            raise passwordError("Password must contain at least one digit.")
        if not any(char in "!@#$%^&*()_-+=/?<>.,:;" for char in password):
            raise passwordError("Password must contain at least one special character.")
        return password
        
    def display(self):
        print(f"Username successfully signed up")
        
    
class SignIn:
    def __init__(self, registered_user):
        # registered_user should be an instance of SignUp
        self.registered_user = registered_user

    def authenticate(self, username, password):
        if username != self.registered_user.username or password != self.registered_user.password:
            raise passwordError("Invalid username or password.")
        print("Sign in successful!")


if __name__ == "__main__":  
    fn = input("Enter first name: ")
    ln = input("Enter last name: ")
    username = input("Enter username: ")
    password = input("Enter password: ")

    try:
        user = SignUp(fn, ln, username, password)
        print("\n--- Sign Up Successful ---")
        user.display()
        
        print("\n--- Sign In ---")
        signin_username = input("Enter username for sign in: ")
        signin_password = input("Enter password for sign in: ")
        
        signin_user = SignIn(user)
        signin_user.authenticate(signin_username, signin_password)
        
    except passwordError as e:
        print(f"Error: {e}")