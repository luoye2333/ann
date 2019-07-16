import os
import sys
print(__file__)
print(os.path.basename(__file__))
print(os.path.dirname(__file__))
print(os.path.abspath(os.path.dirname(__file__)))
print(os.getcwd())
print(sys.argv[0])
print(os.path.basename(sys.argv[0]))
