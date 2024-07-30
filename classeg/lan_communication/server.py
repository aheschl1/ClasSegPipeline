import socket


class ClasSegDataServer:
    def __init__(self, port=3111):
        self.port = port
        self.server_socket = self._instantiate_socket()

    def _instantiate_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind()
        return s

    def run(self):
        ...
