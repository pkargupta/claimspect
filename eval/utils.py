import os
import subprocess

def run_command(command):
    """Executes a shell command and handles errors."""
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")

def run_baseline_evaluation(hierarchy_path, output_directory, baseline_model_name, taxo_height, child_num_per_node, data_dir, topic, claim_id, eval_model):
    """Runs zero-shot and RAG baseline evaluations."""
    baselines = {
        "zeroshot": "eval.baseline.zeroshot_iterative",
        "rag": "eval.baseline.rag_iterative"
    }
    
    for baseline, module in baselines.items():
        output_path = os.path.join(output_directory, f"{baseline_model_name}_{baseline}_hierarchy.json")
        if os.path.exists(output_path):
            print(f"Skipping {baseline} baseline for {hierarchy_path} with existing output at {output_path}")
        else:
            command = [
                "python", "-m", module,
                f"--model_name={baseline_model_name}",
                f"--height={taxo_height}",
                f"--output_path={output_path}",
                f"--input_path={hierarchy_path}",
                f"--data_dir={data_dir}",
                f"--topic={topic}",
                f"--max_aspects_per_node={child_num_per_node}",
                f"--claim_id={claim_id}"
            ]
            run_command(command)

        # Evaluate the baseline results
        run_node_level_evaluation(output_path, output_directory, eval_model, f"{baseline}_baseline_node_level_eval.json")
    
    # Compare the preference of our method with the baselines
    output_path = os.path.join(output_directory, eval_model, f"preference_comparison_our_method.json")
    if os.path.exists(output_path):
        print(f"Skipping preference comparison for {hierarchy_path} with existing output at {output_path}")
    else:
        baseline_json_path_list = [
            os.path.join(output_directory, f"{baseline_model_name}_{baseline}_hierarchy.json")
            for baseline in baselines
        ]
        command = [
            "python", "-m", "eval.taxo_judge.judge",
            f"--baseline_json_path_list={str(baseline_json_path_list).replace('\'', '\"')}",
            f"--method_json_path={hierarchy_path}",
            f'--output_path={output_path}',
            f'--model_name={eval_model}'
        ]    
        run_command(command)
     
    # compare the preference of rag_baseline and zeroshot_baseline
    output_path = os.path.join(output_directory, eval_model, f"preference_comparison_rag.json")
    if os.path.exists(output_path):
        print(f"Skipping preference comparison for {hierarchy_path} with existing output at {output_path}")
    else:
        baseline_json_path_list = [os.path.join(output_directory, f"{baseline_model_name}_zeroshot_hierarchy.json")]
        command = [
            "python", "-m", "eval.taxo_judge.judge",
            f"--baseline_json_path_list={str(baseline_json_path_list).replace('\'', '\"')}",
            f"--method_json_path={os.path.join(output_directory, f'{baseline_model_name}_rag_hierarchy.json')}",
            f'--output_path={output_path}',
            f'--model_name={eval_model}'
        ]
        run_command(command)
    
    
def run_node_level_evaluation(eval_json_path, output_directory, model_judge_name, output_filename):
    """Runs node-level evaluation for a given evaluation JSON path."""
    output_path = os.path.join(output_directory, model_judge_name, output_filename)
    
    if os.path.exists(output_path):
        print(f"Skipping evaluation for {eval_json_path} with existing output at {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        command = [
            "python", "-m", "eval.node_judge.main",
            f"--eval_json_path={eval_json_path}",
            f"--model_judge_name={model_judge_name}",
            f"--output_path={output_path}"
        ]
        run_command(command)

def perform_evaluation(hierarchy_path: str, 
                       output_directory: str, 
                       eval_model: str, 
                       do_eval_node_level: bool,
                       do_eval_taxonomy_level: bool,
                       baseline_model_name: str,
                       taxo_height: int,
                       child_num_per_node: int,
                       data_dir: str,
                       topic: str,
                       claim_id: int):
    """
    Perform evaluation on the data in hierarchy_path and save the results in the output_directory.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Part I: Taxonomy-level comparison
    if do_eval_taxonomy_level:
        run_baseline_evaluation(hierarchy_path, output_directory, baseline_model_name, taxo_height, child_num_per_node, data_dir, topic, claim_id, eval_model)
        
    # Part II: Node-level evaluation
    if do_eval_node_level:
        run_node_level_evaluation(hierarchy_path, output_directory, eval_model, "node_level_eval.json")
