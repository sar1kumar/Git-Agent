"""Sample code with multiple violations for testing the code review agent."""

import os

# SEC001: Hardcoded secrets
password = "super_secret_123"
api_key = "sk-1234567890abcdef"
database_token = "mongodb://admin:password123@localhost:27017"


# S002: Bad naming convention (should be snake_case)
def ProcessUserData(userData):
    """Process user data."""
    return userData


# S002: Bad class naming (already PascalCase, but let's add a bad one)
class user_manager:
    """Manages users."""
    pass


# S001: Line too long (exceeds 120 characters)
very_long_variable_name = "This is an extremely long string that definitely exceeds the maximum line length of 120 characters and should trigger the line length rule"


# SEC002: SQL Injection vulnerability
def get_user_by_id(user_id):
    """Get user by ID - INSECURE!"""
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query


def delete_user(username):
    """Delete user - INSECURE!"""
    query = f"DELETE FROM users WHERE username = '{username}'"
    return query


# BP001: Bare except clause
def risky_operation():
    """Performs a risky operation with bad error handling."""
    try:
        result = 1 / 0
    except:
        pass  # Silently swallowing all exceptions
    return None


# Q001: High cyclomatic complexity (many branches)
def complex_calculator(operation, a, b, c, d, mode):
    """A function with too many branches."""
    result = 0
    
    if operation == "add":
        if mode == "simple":
            result = a + b
        elif mode == "complex":
            result = a + b + c + d
        else:
            result = a + b + c
    elif operation == "subtract":
        if mode == "simple":
            result = a - b
        elif mode == "complex":
            result = a - b - c - d
        else:
            result = a - b - c
    elif operation == "multiply":
        if mode == "simple":
            result = a * b
        elif mode == "complex":
            result = a * b * c * d
        else:
            result = a * b * c
    elif operation == "divide":
        if mode == "simple" and b != 0:
            result = a / b
        elif mode == "complex" and b != 0 and c != 0 and d != 0:
            result = a / b / c / d
        else:
            result = 0
    elif operation == "power":
        if mode == "simple":
            result = a ** b
        elif mode == "complex":
            result = (a ** b) ** c
        else:
            result = a ** 2
    
    return result


# Q002: Function too long (exceeds 50 lines)
def extremely_long_function(data):
    """This function is way too long and should be broken down."""
    # Step 1: Validate input
    if data is None:
        return None
    
    if not isinstance(data, dict):
        return None
    
    # Step 2: Extract fields
    name = data.get("name", "")
    email = data.get("email", "")
    age = data.get("age", 0)
    address = data.get("address", {})
    
    # Step 3: Validate name
    if not name:
        return {"error": "Name is required"}
    
    if len(name) < 2:
        return {"error": "Name too short"}
    
    if len(name) > 100:
        return {"error": "Name too long"}
    
    # Step 4: Validate email
    if not email:
        return {"error": "Email is required"}
    
    if "@" not in email:
        return {"error": "Invalid email format"}
    
    if "." not in email.split("@")[1]:
        return {"error": "Invalid email domain"}
    
    # Step 5: Validate age
    if age < 0:
        return {"error": "Age cannot be negative"}
    
    if age > 150:
        return {"error": "Age seems invalid"}
    
    # Step 6: Process address
    street = address.get("street", "")
    city = address.get("city", "")
    country = address.get("country", "")
    zip_code = address.get("zip", "")
    
    # Step 7: Validate address
    if street and len(street) > 200:
        return {"error": "Street address too long"}
    
    if city and len(city) > 100:
        return {"error": "City name too long"}
    
    # Step 8: Format the data
    formatted_name = name.strip().title()
    formatted_email = email.strip().lower()
    
    # Step 9: Create result
    result = {
        "name": formatted_name,
        "email": formatted_email,
        "age": age,
        "address": {
            "street": street,
            "city": city,
            "country": country,
            "zip": zip_code,
        },
        "is_valid": True,
        "processed_at": "2024-01-01",
    }
    
    # Step 10: Return
    return result


# Another BP001: Broad exception with pass
def another_bad_handler():
    """Another example of bad error handling."""
    try:
        open("nonexistent_file.txt")
    except Exception:
        pass