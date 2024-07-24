import os
import sys


def find_project_root(project_name):
    # Start with the current working directory
    current_path = os.getcwd()

    while True:
        # Check if the current directory's name matches the project name
        if os.path.basename(current_path) == project_name:
            return current_path

        # Move up one level in the directory tree
        parent_path = os.path.dirname(current_path)

        # If we've reached the root directory, we stop the search
        if parent_path == current_path:
            raise FileNotFoundError(f"Project root '{project_name}' not found.")

        # Update the current path to the parent path for the next iteration
        current_path = parent_path


def update_env_file(project_root_path, key, value):
    env_file_path = os.path.join(project_root_path, ".env")
    lines = []
    key_found = False
    quoted_value = f'"{value}"\n'
    path_updated = False

    # Read existing .env file if it exists
    if os.path.exists(env_file_path):
        with open(env_file_path, "r") as file:
            lines = file.readlines()

        # Update the value if the key exists
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={quoted_value}"
                key_found = True
            elif line.startswith("PATH="):
                # Append the project root path to the PATH variable
                path_value = line[len("PATH=") :].strip()
                if project_root_path not in path_value:
                    if path_value.endswith('"'):
                        path_value = path_value[:-1]
                    lines[i] = f'PATH="{path_value}:{value}"\n'
                    path_updated = True
            if key_found and path_updated:
                break

    # If key wasn't found, add it to the end
    if not key_found:
        # Ensure there is a newline before appending the new entry if file is not empty
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={quoted_value}")

    # If PATH wasn't updated, add it to the end
    if not path_updated:
        # Ensure there is a newline before appending the new entry if file is not empty
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f'PATH="{value}:$PATH"\n')

    # Write the updated lines back to the .env file
    with open(env_file_path, "w") as file:
        file.writelines(lines)


# Example usage
project_name = "mlops_project"
env_key = "PROJECT_PATH"

try:
    project_root_path = find_project_root(project_name)
    print(f"Project root path: {project_root_path}")
    update_env_file(project_root_path, env_key, project_root_path)
    print(f'Updated .env file with {env_key}="{project_root_path}" and added path to PATH')
except FileNotFoundError as e:
    print(e)
