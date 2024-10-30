import time
import sys

print("foo")
print(sys.argv)

for i in range(5):
    time.sleep(1)
    print(i)