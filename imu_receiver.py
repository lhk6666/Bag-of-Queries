import serial

SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 921600
UDP_IP = '192.168.xx.xx'  # 目标主机IP
UDP_PORT = 12345

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)

print("Serial to UDP bridge started.")

while True:
    data = ser.read(ser.in_waiting or 1)
    print(data)

