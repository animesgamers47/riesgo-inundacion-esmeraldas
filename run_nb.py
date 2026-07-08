"""Execute notebook with jupyter nbconvert"""
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
lines = result.stdout.split('\n')
if len(lines) > 5:
    print('\n'.join(lines[-5:]))
else:
    print(result.stdout[-500:])
if result.stderr:
    err_lines = result.stderr.split('\n')
    non_joblib = [l for l in err_lines if 'joblib' not in l and 'resource_tracker' not in l]
    if non_joblib:
        print('\n'.join(non_joblib[:10]))
print(f"Return code: {result.returncode}")
