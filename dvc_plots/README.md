This directory is where DVC will place html plots, as an `index.html` file.
The file is large and would bloat the git repo, so it's better to recreate on the fly or have DVC track the file.

see https://dvc.org/doc/command-reference/plots

"We recommend to track these source image files with DVC instead of Git, to prevent the repository from bloating."
