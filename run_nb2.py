"""Execute notebook and show errors"""
import subprocess, os
os.chdir(r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico")
cmd = [
    "python", "-m", "nbconvert",
    "--to", "notebook",
    "--execute", "Proyecto_2P_Riesgo_Inundacion.ipynb",
    "--output", "Proyecto_2P_Riesgo_Inundacion.ipynb",
    "--ExecutePreprocessor.timeout=600"
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print("=== STDOUT ===")
print(result.stdout[-2000:])
print("\n=== STDERR (filtered) ===")
for line in result.stderr.split('\n'):
    if 'CellExecutionError' in line or 'Error:' in line or 'SyntaxError' in line or 'traceback' in line.lower():
        print(line)
    elif 'ERROR' in line:
        print(line)
print(f"\nReturn code: {result.returncode}")
