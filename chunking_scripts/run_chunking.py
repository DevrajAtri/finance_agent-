import json
import subprocess
import sys
import os
import argparse

def run_task(task: dict):
    """
    Runs a single chunking task defined in the config.
    """
    task_name = task['name']
    script_path = task['script_path']
    input_dir = task['input_dir']
    output_dir = task['output_dir']
    
    print(f"\n--- Running Task: {task_name} ---")
    
    # 1. Check if the script file actually exists
    if not os.path.exists(script_path):
        print(f"FAILED: Script not found at {script_path}. Please check 'chunking_config.json'.")
        return

    # 2. Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"FAILED: Input directory not found at {input_dir}. Skipping task.")
        return
        
    # 3. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Get the python executable from the current virtual environment
    # This ensures we use the Python from '.venv' with all our packages
    python_exe = sys.executable 
    
    # 5. Build the command to run the script
    command = [
        python_exe,
        script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir
    ]
    
    print(f"Executing: {' '.join(command)}")
    
    try:
        # 6. Run the command
        # We use 'check=True' so it will raise an error if the chunking script fails
        subprocess.run(command, check=True)
        print(f"Success: {task_name}")
        
    except subprocess.CalledProcessError as e:
        print(f" FAILED: {task_name} returned an error.")
        print(f"Error: {e}\n")
    except FileNotFoundError:
        print(f" FAILED: Could not find '{python_exe}' or '{script_path}'.")

def main():
    """
    Main function to load config and run all tasks.
    """
    # Load the config file
    config_path = "chunking_config.json"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Run each task defined in the config
    if 'tasks' not in config:
        print("Error: Config file is missing the 'tasks' list.")
        return
        
    for task in config['tasks']:
        run_task(task)
    
    print("\n--- All chunking tasks complete. ---")

if __name__ == "__main__":
    main()