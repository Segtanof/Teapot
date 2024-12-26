# filename: fibonacci.py

def fibonacci(n):
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]

print(f"First 10 numbers of the Fibonacci sequence are: {fibonacci(10)}")