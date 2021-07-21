import sys
from pathlib import Path
from shutil import copytree, ignore_patterns


# This script initializes new pytorch project with the template files.
# Run `python3 new_project.py ../NewProject` then a new project named
# NewProject will be created

current_dir = Path()
assert (current_dir / 'copy_project.py').is_file(
), 'Script should be executed in the pytorch-project-template directory'
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: python3 copy_project.py DIR_PATH_TO_NEW_PROJECT'

project_name = Path(sys.argv[1])
target_dir = current_dir / project_name

ignore = [".git", "data", "copy_project.py", "venv*",
          "logs*", "checkpoints*", "experiments*",
          "__pycache__", ".DS_Store"]

copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore))
print('New project initialized at', target_dir.absolute().resolve())
