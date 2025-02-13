import os
import subprocess

def perform_evaluation(hierarchy_path: str, output_directory: str, eval_model: str):
    """
    Perform evaluation on the data in hierarchy_path and save the results in the output_directory.
    """
    
    # Implementation of evaluation logic
    eval_json_path = hierarchy_path
    model_judge_name = eval_model
    output_path = os.path.join(output_directory, eval_model, "node_level_eval.json")
    
    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Execute the command
    command = [
        "python", "-m", "eval.node_judge.main",
        f"--eval_json_path={eval_json_path}",
        f"--model_judge_name={model_judge_name}",
        f"--output_path={output_path}"
    ]
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")
