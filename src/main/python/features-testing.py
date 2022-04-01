
from pathlib import Path

f = open(str(Path('main.py')), mode='r')
content = f.read()
print(content)
f.close()
