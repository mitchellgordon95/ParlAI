import socket
import time
import re

def netcat(hostname, port, content):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))
    s.settimeout(0.5)
    s.sendall(content.encode("utf-8"))
    out = ""
    while 1:
        try:
            data = s.recv(1024)
        except socket.timeout:
            break
        else:
            out += data.decode('utf-8')
    s.shutdown(socket.SHUT_WR)
    s.close()
    return out

resp = netcat("0.tcp.ngrok.io", 13396, "hi\n")
print(resp)
print(re.findall(r'\[text_1\]: (.*)', resp)[-1])
