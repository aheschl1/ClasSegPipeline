import subprocess

TERMINALS = {
    "gnome-terminal": "gnome-terminal -- bash -c '{command}; exec bash'",
    "konsole": "konsole -e bash -c '{command}; exec bash'",
    "xterm": "xterm -hold -e '{command}'",
    "xfce4-terminal": "xfce4-terminal -e '{command}'",
    "mate-terminal": "mate-terminal -e '{command}'",
    "terminator": "terminator -e '{command}'"
}


def get_terminal_command(command):
    for terminal in TERMINALS:
        if subprocess.call(f'command -v {terminal}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
            return TERMINALS[terminal].format(command=command)
    return None
