# Schottky-Circles-IIS-algorithm
python code to generate schottky circles based on an initial configuration

## Setup Locally

First ensure that you have python and pip installed on your system, you can verify this with:
- `python --version` (or python3)
- `pip --version`

Commands:
- `git clone https://github.com/hllx808/Schottky-Circles-IIS-algorithm.git` (Copy the project to your system)
- `cd Schottky-Circles-IIS-algorithm` (Enter the project Directory)
- `python3 -m venv .venv` (Creates an environment specific to this project)
- `source ./.venv/bin/activate` (Activates said environment in your terminal)
- `pip install -r requirements.txt` (Installs the necessary packages)
- `python3 circles_inverted.py` (runs the program)

Running on VS Code:
- First Ensure that Python is installed and working on VS Code, Python on your system and the Python Extension on VSCode should be everything you need to run files on their app.
- To run this python file **on the `.venv` environment** that you've created you must first open the `command pallete` 
- Then search for: "Select Interpreter" and choose the Virtual Environment that you've created (`.venv`). 

## Google Colab Link:
- [Create a copy of colab to modify existing circle configuration](https://colab.research.google.com/drive/1BX3LGmTZ28EEG6q2qKE-IAhv8IMiHn7f?usp=sharing)

## Credits:
- [Soma-Arc and their team's Iterated Inversion System Algorithm](https://archive.bridgesmathart.org/2016/bridges2016-367.html)
- Indra's Pearls by David Mumford, Caroline Series and David Wright

## Figures:

### Apollonian Gasket
![apollonian gasket](./images/apollonian%20gasket.png)
