import sys
import json

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(filetypes=[("Hist files", ".tiff")])

print(json.dumps(file_path))
sys.stdout.flush()