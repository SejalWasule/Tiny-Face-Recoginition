import os
import subprocess
import sys

def print_menu():
    print("\n" + "="*40)
    print("   Face Recognition System Menu")
    print("="*40)
    print("1. Create Dataset (1_datasetCreation.py)")
    print("2. Preprocess Embeddings (2_preprocessingEmbeddings.py)")
    print("3. Train Face ML Model (3_trainingFaceML.py)")
    print("4. Recognize Person (4_recognizingPerson.py)")
    print("5. Recognize Person with CSV (5_recognizingPersonwithCSVDatabase.py)")
    print("6. Exit")
    print("="*40)

def run_script(script_name):
    """Runs a python script in a subprocess."""
    script_path = os.path.join(os.getcwd(), script_name)
    if not os.path.exists(script_path):
        print(f"[ERROR] File not found: {script_name}")
        return

    print(f"\n[INFO] Starting {script_name}...")
    try:
        # Use sys.executable to ensure we use the same python interpreter
        result = subprocess.run([sys.executable, script_name], check=False)
        if result.returncode == 0:
            print(f"[INFO] {script_name} finished successfully.")
        else:
            print(f"[WARNING] {script_name} exited with code {result.returncode}.")
    except Exception as e:
        print(f"[ERROR] Failed to run {script_name}: {e}")

def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            run_script("1_datasetCreation.py")
        elif choice == '2':
            run_script("2_preprocessingEmbeddings.py")
        elif choice == '3':
            run_script("3_trainingFaceML.py")
        elif choice == '4':
            run_script("4_recognizingPerson.py")
        elif choice == '5':
            run_script("5_recognizingPersonwithCSVDatabase.py")
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("[ERROR] Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
