import logging
import socket
import psutil


def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
    except Exception as e:
        logging.log(logging.ERROR, f"Error getting LAN IP: {e}")
        raise e
    finally:
        s.close()
    return lan_ip


class ClasSegLANClient:
    def __init__(self, port):
        self._port = port
        self._ip = get_lan_ip()
        self._broadcast_address, self._netmask = self._get_broadcast_address()
        self._validate_address_and_broadcast()

    def _validate_address_and_broadcast(self):
        """
        Ensure that the IP and broadcast address match where the netmask is 255.
        Returns:
        """
        if not self._broadcast_address:
            raise ValueError("Could not find broadcast address.")
        if not self._netmask:
            raise ValueError("Could not find netmask.")
        # ensure that the ip matches the broadcast where the mask is 255
        ip_parts = self._ip.split(".")
        broadcast_parts = self._broadcast_address.split(".")
        netmask_parts = self._netmask.split(".")
        for i, b, m in zip(ip_parts, broadcast_parts, netmask_parts):
            if m == "255" and i != b:
                raise ValueError("IP and broadcast address do not match.")

    def _get_broadcast_address(self):
        """
        Get the broadcast address and netmask for the current IP.
        Returns: Tuple of broadcast address and netmask.
        """
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET and addr.address == self._ip:
                    return addr.broadcast, addr.netmask
        return None, None


if __name__ == "__main__":
    client = ClasSegLANClient(3111)
    print(client._ip)
    print(client._broadcast_address)
    print(client._netmask)
    # client.run()
