import pandas as pd
import subprocess
import os
import psutil
import platform
import time  # === CHANGED ===

def is_license_server_running():
    """
    Checks if LIWC license server is running, cross-platform.
    """
    system = platform.system()
    
    if system == "Windows":
        process_name = "LIWC-22-license-server.exe"
    elif system == "Darwin":  # macOS
        process_name = "LIWC-22-license-server"  # the macOS binary name
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if proc.info['name'] == process_name:
            return True
    return False


def start_liwc_license_server():
    """
    Launches the LIWC license server on Windows or macOS.
    """
    system = platform.system()
    
    if system == "Windows":
        license_server_path = r"C:\Program Files\LIWC-22\LIWC-22-license-server\LIWC-22-license-server.exe"
    elif system == "Darwin":  # macOS
        # === CHANGED ===
        license_server_path = "/Applications/LIWC-22.app"
    else:
        print(f"Unsupported operating system: {system}")
        return

    if is_license_server_running():
        print("LIWC license server is already running.")
        return

    if not os.path.exists(license_server_path):
        print(f"License server not found at {license_server_path}. Double-check installation path.")
        return

    try:
        # === CHANGED ===
        if system == "Darwin":
            subprocess.Popen(["open", "-a", license_server_path])
            # Wait up to 10 seconds for the server to start
            for _ in range(10):
                if is_license_server_running():
                    break
                time.sleep(1)
        else:
            subprocess.Popen([license_server_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # === END CHANGED ===
        print("LIWC license server launched successfully.")
    except Exception as e:
        print(f"Failed to launch license server: {e}")


def get_liwc_cli_command(input_path, column_index, output_path, custom_dictionary=None):
    """
    Returns the correct LIWC CLI command for Windows or macOS.
    """
    system = platform.system()
    
    if system == "Windows":
        cli_executable = "LIWC-22-cli.exe"
    elif system == "Darwin":
        # === CHANGED ===
        cli_executable = "/Applications/LIWC-22.app/Contents/MacOS/LIWC-22-cli"
        # === END CHANGED ===
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    cmd = [
        cli_executable,
        "--mode", "wc",
        "--input", input_path,
        "--column-indices", column_index,
        "--output", output_path
    ]
    if custom_dictionary:
        cmd += ["--dictionary", custom_dictionary]
    return cmd


def classify_liwc(file: str, column: str, dependent: bool = False, merge_back: bool = False,
                  concise: bool = False, custom_dictionary: str = None):
    """
    Run LIWC twice — default + optional custom dictionary — and merge results.
    """
    print("Running LIWC analysis...")
    start_liwc_license_server()

    input_path = file
    input_dir = os.path.dirname(input_path) or "."
    
    output_file_default = "liwc_captions.csv" if dependent else "total_linguistic_analysis.csv"
    output_file_custom = "liwc_captions_custom.csv" if dependent else "total_linguistic_analysis_custom.csv"
    merged_output_file = "liwc_combined.csv"

    output_path_default = os.path.join(input_dir, output_file_default)
    output_path_custom = os.path.join(input_dir, output_file_custom)
    merged_output_path = os.path.join(input_dir, merged_output_file)

    df = pd.read_csv(input_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in file.")
    col_index = str(df.columns.get_loc(column) + 1)  # 1-based for LIWC CLI

    original_df = df.copy()

    # --- First Run: Default Dictionary ---
    try:
        cmd_default = get_liwc_cli_command(input_path, col_index, output_path_default)
        print("\nRunning LIWC with default dictionary...")
        subprocess.run(cmd_default, check=True, cwd=input_dir)
        print("Default LIWC analysis completed successfully.")

        liwc_df_default = pd.read_csv(output_path_default)
        if concise:
            columns_to_keep = ["achieve", "Affect", "affiliation", "Analytic",
                               "article", "Authentic", "cogproc", "differ", "emo_anx", "emo_sad", "exclusive",
                               "filler", "function", "home", "motion", "Drives",
                               'reward', 'risk', 'curiosity', "negate", "number", "Perception",
                               "we", "i", "you", "relig", "emo_pos", "emo_neg", "tone_pos",
                               "tone_neg", "Social", "space", "swear", "time", "work"]
            liwc_df_default = liwc_df_default[[col for col in columns_to_keep if col in liwc_df_default.columns]]

    except subprocess.CalledProcessError as e:
        print(f"Error running LIWC default analysis: {e}")
        return

    # --- Second Run: Custom Dictionary ---
    liwc_df_custom = None
    if custom_dictionary:
        if not os.path.exists(custom_dictionary):
            raise FileNotFoundError(f"Custom dictionary file not found: {custom_dictionary}")

        try:
            cmd_custom = get_liwc_cli_command(input_path, col_index, output_path_custom, custom_dictionary)
            print("\nRunning LIWC with custom dictionary...")
            subprocess.run(cmd_custom, check=True, cwd=input_dir)
            print("Custom LIWC analysis completed successfully.")

            liwc_df_custom = pd.read_csv(output_path_custom)
            liwc_df_custom = liwc_df_custom.add_suffix("_custom")

        except subprocess.CalledProcessError as e:
            print(f"Error running LIWC custom analysis: {e}")

    # --- Merge Back Results ---
    if merge_back:
        print("\nMerging LIWC outputs back into original file...")
        merged_df = pd.concat([original_df.reset_index(drop=True),
                               liwc_df_default.reset_index(drop=True)], axis=1)
        if liwc_df_custom is not None:
            merged_df = pd.concat([merged_df.reset_index(drop=True),
                                   liwc_df_custom.reset_index(drop=True)], axis=1)

        merged_df.to_csv(merged_output_path, index=False)
        print(f"Combined LIWC output written to: {merged_output_path}")
        print(list(merged_df.columns))
        return merged_df

