"""
for NFS, it takes several minutes to load zero123 weight from disk to mem. 
So I cache it in mem and sue IPC-socket to fetch wh
"""
import socket
import psutil

def print_mem_occupied_by_me():
    
    pid = psutil.Process().pid

    
    process = psutil.Process(pid)
    mem_info = process.memory_info()

    
    print(f"Memory occupied by current process (PID {pid}):")
    print(f"RSS (Resident Set Size): {mem_info.rss} bytes {mem_info.rss/1024/1024}  MB {mem_info.rss/1024/1024/1024} GB")
    print(f"VMS (Virtual Memory Size): {mem_info.vms} bytes {mem_info.vms/1024/1024}  MB {mem_info.vms/1024/1024/1024} GB")


class MemoryCache:
    PORT = 40639
    bytes_ = None
    BYTES_OF_DATASIZE=639 
    @classmethod
    def send_dataSize(cls,client_socket):
        
        size = len(cls.bytes_)
        print(f"dataSize = {size} B")
        size_bytes = size.to_bytes(cls.BYTES_OF_DATASIZE, byteorder='big')
        client_socket.sendall(size_bytes)
    @classmethod
    def recv_dataSize(cls, server_socket)->int:
        
        size_bytes = server_socket.recv(cls.BYTES_OF_DATASIZE)
        size = int.from_bytes(size_bytes, byteorder='big')
        print(f"dataSize = {size} B")
        return size
    @classmethod
    def run_as_server(cls, path_fileToKeepInMemory: str):
        """
        From 1024 to 49151: These ports are known as the Registered ports. These ports can be used by ordinary user processes or programs executed by ordinary users.
        From 49152 to 65535: These ports are known as Dynamic Ports.
        """

        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('localhost', cls.PORT))
        server_socket.listen(1)
        #
        print('reading...')
        with open(path_fileToKeepInMemory, 'rb') as f:
            cls.bytes_ = f.read()
        print('read over')
        print_mem_occupied_by_me()
        while True:
            print('等待连接...')
            client_socket, _ = server_socket.accept()
            try:
                
                print('有客户端连接.sending...')
                cls.send_dataSize(client_socket)
                
                client_socket.sendall(cls.bytes_)  # eg. b'1010111', f.read()
                print('send over')
            except Exception as e:
                print(f"e=",e)
            finally:
                
                client_socket.close()
    @classmethod
    def receive(cls)->bytes:
        
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost',  cls.PORT))

        
        SIZE=cls.recv_dataSize(client_socket)
        model_data = bytearray(SIZE)
        bytes_received = 0

        while bytes_received < SIZE:
            data = client_socket.recv(SIZE - bytes_received)
            if not data:
                break
            model_data[bytes_received:bytes_received + len(data)] = data 
            bytes_received += len(data)
        model_data = bytes(model_data)
        print(f"Received {len(model_data)/(1024*1024)} MB, {len(model_data)/(1024*1024*1024)} GB")
        assert len(model_data)==SIZE
        print('len(model_data)==SIZE,恭喜！')
        
        client_socket.close()
        return model_data
if __name__ == '__main__':
    import sys,os
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(cur_dir)
    sys.path.append(os.path.join(cur_dir, ".."))
    import root_config
    MemoryCache.run_as_server(path_fileToKeepInMemory=root_config.weightPath_zero123)