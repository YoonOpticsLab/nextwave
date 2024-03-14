#  listener_thread = Thread(target=pyshmem.do_listen, args=[socket_callback] )
#  listener_thread.daemon=True # So application will terminate even if this thread is alive
#  listener_thread.start()    
from threading import Thread
import socket
import os
import time

WINDOWS=(os.name == 'nt')

# Make sure these match the C++ Code
SOCKET_CAMERA=50007
SOCKET_CENTROIDING=50008

class SocketComponent():
    def __init__(self,port):
        self.port=port
        self.s = None

    def init(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("localhost", self.port))

    def send(self,msg):
        if not self.s is None:
            self.s.send(msg)

class NextwaveSocketComm():
    """ Class to manage:
          - Structures needed for realtime engine (boxes/refs, computed centroids, etc.)
          - Communication with the realtime engine (comm. over shared memory)
          - Computation of Zernikes, matrices for SVD, etc.
    """
    def __init__(self,ui):
        # TODO:
        self.ui = ui

    def init(self):
        self.camera = SocketComponent(SOCKET_CAMERA)
        self.camera.init()
        time.sleep(0.5)
        self.centroiding = SocketComponent(SOCKET_CENTROIDING)
        self.centroiding.init()
        #os.sleep(0.5)

def do_listen(fn_callback, port):
   #fn_callback=args[0]

   listen_address=('localhost',port); #('127.0.0.1',27015);
   done=False

   while done==False:# Allow connect/reconnect forever

     print('Waiting for connection')

     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # prevent address in use error
        s.bind(listen_address)
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while done==False:
                data = conn.recv(1024) # Don't think this number matters too much

                if len(data)==0:
                    time.sleep(0.1)
                    continue
                    
                print(data)

                if data==b'reset':
                    done=False
                    break
          
                elif data==b'quit':
                    done=True
                    sys.exit()
                    break
                    
                elif b'next' in data:
                    amt=int(data[5:])
                    fn_callback('next',amt)
                    
                elif data[0:4]==b"send":
                    str_dim=data[5:]
                    dims=str_dim.split(b',');
                    dims = np.array( [int(dim1) for dim1 in dims] )
                    #print (dims)

                    size_single = 4
                    size_total=np.prod(dims)*size_single
                    
                    shmem = mmap.mmap(-1, size_total ,"shm")
                    shmem.seek(0)
                    buf = shmem.read(size_total)
                    data = np.frombuffer(buf, dtype=np.float32).reshape(dims)
                    shmem.close()
                    
                    fn_callback('data',data)
                    
                    break
            
     
