# Simple script to read and print the first 10 lines of the entityVector.txt file

file_path = '/home/e706/zhanxiangning/KG/transE/entityVector.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as f:  # utf-8 to handle possible Chinese characters
        for i in range(10):
            line = f.readline()
            if not line:  # End of file reached before 10 lines
                break
            print(f"Line {i+1}: {line.rstrip()}")  # rstrip() removes trailing newline/spaces
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    print("Please check the path and make sure the file exists.")
except Exception as e:
    print(f"An error occurred: {e}")