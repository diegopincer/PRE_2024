import socket

# Configurações do servidor UDP
UDP_IP = "172.43.100.100"  # Endereço IP do servidor
UDP_PORT = 5005       # Porta do servidor

# Criar o socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Servidor UDP iniciado em {UDP_IP}:{UDP_PORT}")

while True:
    # Receber mensagem do cliente
    data, addr = sock.recvfrom(1024)  # Buffer de 1024 bytes
    message = data.decode()
    print(f"Mensagem recebida de {addr}: {message}")

    # Processar a mensagem recebida
    delta_x, delta_y = message.split(',')
    delta_x = int(delta_x)
    delta_y = int(delta_y)
    
    # Exibir os valores de delta
    print(f"Delta X: {delta_x}, Delta Y: {delta_y}")
