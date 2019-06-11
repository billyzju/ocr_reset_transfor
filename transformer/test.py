import sys
import os


name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(name)
sys.path.append(name)
import HyperParameters as hp

print(hp.EPOCH)
