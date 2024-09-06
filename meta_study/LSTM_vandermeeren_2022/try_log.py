import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # Save the original stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)  # Write to the console (terminal)
        self.log.write(message)       # Write to the file

    def flush(self):
        pass  # Python requires this for compatibility

# Redirect all stdout (print) to both console and file
sys.stdout = Logger("output.txt")

# Your code with print statements
print("This will go to both console and file.")
print("Another line that will be written to both.")

# At the end of the script, you can close the file if needed
sys.stdout.log.close()
