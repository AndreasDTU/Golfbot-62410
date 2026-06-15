import socket
import sys

def connect(host, port=5555):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(15.0)
    s.connect((host, port))
    return s

def send(sock, cmd):
    sock.sendall((cmd + '\n').encode('utf-8'))
    return sock.recv(1024).decode('utf-8').strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pc.py <EV3_IP>")
        sys.exit(1)

    host = sys.argv[1]
    print("Connecting to {}...".format(host))
    sock = connect(host)
    print("Connected!\n")

    while True:
        try:
            raw = input("ev3> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        # --- Local commands ---
        if cmd == 'quit':
            print("\nExiting...")
            break

        # --- Robot commands (forwarded over socket) ---
        response = send(sock, raw)
        print(response)

    sock.close()

if __name__ == '__main__':
    main()
