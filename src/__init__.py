import sys
import os

# Get the current directory (project root) and add 'src' to the path
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)
