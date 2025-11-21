import logging
import os
import re


def save_workflow_json(workflow_json: str, filename_prefix: str, output_dir: str) -> str:
    """
    Save workflow JSON to a file with smart incrementing counter.

    Args:
        workflow_json: The workflow JSON string to save.
        filename_prefix: The filename prefix or path.
        output_dir: The base output directory.

    Returns:
        The full path to the saved file.
    """
    try:
        original_input = filename_prefix.strip()
        ends_with_slash = original_input.endswith("/") or original_input.endswith("\\")

        normalized_input = original_input.replace("\\", "/").lstrip("/")
        normalized_input = normalized_input.rstrip("/")

        if "/" in normalized_input:
            parts = normalized_input.split("/")
            parts = [p for p in parts if p]
            if len(parts) > 1:
                subdirectory = "/".join(parts[:-1])
                base_filename = parts[-1]
            else:
                subdirectory = parts[0] if parts else ""
                base_filename = "generated_workflow"
        else:
            if ends_with_slash or not normalized_input:
                subdirectory = normalized_input if normalized_input else ""
                base_filename = "generated_workflow"
            else:
                subdirectory = ""
                base_filename = normalized_input

        if base_filename.endswith(".workflow.json"):
            base_filename = base_filename[:-15]
        elif base_filename.endswith(".json"):
            base_filename = base_filename[:-5]

        if not base_filename:
            base_filename = "generated_workflow"

        full_output_folder = os.path.join(output_dir, subdirectory) if subdirectory else output_dir

        if not os.path.exists(full_output_folder):
            try:
                os.makedirs(full_output_folder, exist_ok=True)
            except Exception as e:
                logging.warning(f"Failed to create directory {full_output_folder}: {e}")

        counter = 1
        if os.path.exists(full_output_folder):
            file_pattern = re.compile(rf"^{re.escape(base_filename)}_(\d+)\.json$")

            max_counter = 0
            try:
                for file in os.listdir(full_output_folder):
                    match = file_pattern.match(file)
                    if match:
                        try:
                            cnt = int(match.group(1))
                            if cnt > max_counter:
                                max_counter = cnt
                        except ValueError:
                            pass
            except Exception as e:
                logging.warning(f"Failed to scan output directory for counter: {e}")

            if max_counter > 0:
                counter = max_counter + 1

        file_path = os.path.join(full_output_folder, f"{base_filename}_{counter:03}.json")

        with open(file_path, "w") as f:
            f.write(workflow_json)

        return file_path

    except Exception as e:
        logging.warning(f"Failed to save workflow: {e}")
        raise e
