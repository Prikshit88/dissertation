import os
import subprocess
import sys
import venv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
output_dir = r"C:\Users\ptush\OneDrive\Documents\Dessertation\output"
os.makedirs(output_dir, exist_ok=True)

# 1. Create virtual environment if not exists
venv_dir = os.path.join(os.getcwd(), "venv")
if not os.path.exists(venv_dir):
    print("Creating virtual environment...")
    venv.create(venv_dir, with_pip=True)
else:
    print("Virtual environment already exists.")

# 2. Define pip and python executables for the venv
if os.name == "nt":
    python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
else:
    python_exe = os.path.join(venv_dir, "bin", "python")
    pip_exe = os.path.join(venv_dir, "bin", "pip")

# 3. Install required packages
requirements = [
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "xgboost"
]
print("Installing required packages...")
subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.check_call([pip_exe, "install"] + requirements)

# 4. Run scripts in sequence
scripts = ["detection.py", "XGBoost.py", "SVM2.py"]
for script in scripts:
    print(f"\nRunning {script} ...")
    result = subprocess.run([python_exe, script], cwd=os.getcwd())
    if result.returncode != 0:
        print(f"Error: {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"{script} completed successfully.")

print("\nAll scripts executed successfully!")