# Injector_Guide.py: Interactive Python and Injector guide
# Designed for Skulpt 1.2.0, compatible with Injector HTML

import random
import re

def print_menu():
    """Display the main menu."""
    print("Injector Guide: Welcome to the Interactive Python & Injector Guide!")
    print("Available commands:")
    print("  python_basics   - Learn Python fundamentals")
    print("  injector_usage  - How to use the Injector app")
    print("  injector_how    - How Injector works")
    print("  lesson_1       - Lesson: Variables and Data Types")
    print("  lesson_2       - Lesson: Control Structures")
    print("  project_calc   - Mini-Project: Build a Calculator")
    print("  project_guess  - Mini-Project: Number Guessing Game")
    print("  help           - Show this menu")
    print("  exit           - Exit the guide")
    print("Type a command and press Enter.")

def python_basics():
    """Teach Python fundamentals."""
    print("Injector Guide: Python Basics")
    print("- Python is a high-level, interpreted programming language.")
    print("- Key features: readable syntax, dynamic typing, extensive libraries.")
    print("- In Injector, Python runs via Skulpt, supporting basic Python 3.x.")
    print("- Data types: int (e.g., 42), float (e.g., 3.14), str (e.g., 'hello'), bool (e.g., True).")
    print("- Example: Try injecting this code in Injector:")
    print("  x = 5; y = 'Hello'; print(x, y)  # Outputs: 5 Hello")
    print("Use 'lesson_1' for more on variables and data types.")

def injector_usage():
    """Explain how to use Injector."""
    print("Injector Guide: How to Use Injector")
    print("- Injector is a browser-based Python console using Skulpt 1.2.0.")
    print("- Features:")
    print("  - Execute: Run single-line Python code in the text input.")
    print("  - Inject: Open popup to paste multi-line scripts.")
    print("  - Eject: Clear console and reset state.")
    print("  - Troubleshoot: Analyze code for errors.")
    print("- Example: Type `print('Hello, Injector!')` in the text input and click Execute.")
    print("- To inject a script, click Inject, paste code, and click Inject Script.")
    print("- Tip: Use 'eject' to reset if errors occur.")

def injector_how():
    """Explain how Injector works."""
    print("Injector Guide: How Injector Works")
    print("- Injector uses Skulpt to run Python in the browser.")
    print("- Skulpt translates Python to JavaScript for execution.")
    print("- Supported modules: random, re. Others (e.g., os) are unavailable.")
    print("- Input/Output: Uses async input() for user interaction; output appears in the console.")
    print("- Error handling: Troubleshoot button catches syntax errors, undefined variables, etc.")
    print("- Example error: `import os` will fail due to module restrictions.")
    print("- UI: HTML/CSS for console and popup; JavaScript handles Skulpt integration.")

def lesson_1():
    """Lesson on variables and data types."""
    print("Injector Guide: Lesson 1 - Variables and Data Types")
    print("- Variables store data; no type declaration needed.")
    print("- Example:")
    print("  name = 'Alice'  # String")
    print("  age = 25       # Integer")
    print("  height = 5.7   # Float")
    print("  is_student = True  # Boolean")
    print("- Try this in Injector:")
    print("  x = 10; y = 20; print(x + y)  # Outputs: 30")
    print("Exercise: Inject the above code, then modify it to print your name and age.")

def lesson_2():
    """Lesson on control structures."""
    print("Injector Guide: Lesson 2 - Control Structures")
    print("- If statements control flow based on conditions.")
    print("- Loops (for, while) repeat code.")
    print("- Example:")
    print("  for i in range(3):")
    print("      print('Loop', i)")
    print("  # Outputs: Loop 0, Loop 1, Loop 2")
    print("- Try this in Injector:")
    print("  x = 5")
    print("  if x > 0:")
    print("      print('Positive')")
    print("Exercise: Inject the above code, then add an else clause to print 'Non-positive'.")

def project_calculator():
    """Mini-project: Build a calculator."""
    print("Injector Guide: Mini-Project - Calculator")
    print("Try injecting this code to build a simple calculator:")
    print("""
def calculator():
    print("Simple Calculator")
    a = float(input("Enter first number: "))
    op = input("Enter operator (+, -, *, /): ")
    b = float(input("Enter second number: "))
    if op == '+':
        print(a + b)
    elif op == '-':
        print(a - b)
    elif op == '*':
        print(a * b)
    elif op == '/':
        print(a / b if b != 0 else 'Error: Division by zero')
    else:
        print("Invalid operator")
calculator()
    """)
    print("Steps: Click Inject, paste the code, click Inject Script, then follow prompts.")

def project_guessing_game():
    """Mini-project: Number guessing game."""
    print("Injector Guide: Mini-Project - Number Guessing Game")
    print("Try injecting this code to build a guessing game:")
    print("""
import random
def guessing_game():
    number = random.randint(1, 10)
    print("Guess a number between 1 and 10!")
    while True:
        guess = int(input("Enter your guess: "))
        if guess == number:
            print("Correct! You won!")
            break
        elif guess < number:
            print("Too low, try again!")
        else:
            print("Too high, try again!")
guessing_game()
    """)
    print("Steps: Click Inject, paste the code, click Inject Script, then guess the number.")

def main():
    """Run the interactive guide."""
    print_menu()
    while True:
        try:
            command = input("Enter command: ")
            command = command.strip().lower()
            if command == "exit":
                print("Injector Guide: Exiting. Thanks for learning!")
                break
            elif command == "help":
                print_menu()
            elif command == "python_basics":
                python_basics()
            elif command == "injector_usage":
                injector_usage()
            elif command == "injector_how":
                injector_how()
            elif command == "lesson_1":
                lesson_1()
            elif command == "lesson_2":
                lesson_2()
            elif command == "project_calc":
                project_calculator()
            elif command == "project_guess":
                project_guessing_game()
            else:
                print(f"Injector Guide: Invalid command '{command}'. Type 'help' for options.")
        except Exception as e:
            print(f"Injector Guide: Error - {str(e)}. Type 'help' for commands.")

if __name__ == "__main__":
    main()
