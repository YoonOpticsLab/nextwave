import numpy as np
import sys
import os
import time

import mmap
import struct
import extract_memory

WINDOWS=(os.name == 'nt')

MEM_LEN=512
MEM_LEN_DATA=2048*2048*4

class ByteStream(bytearray):
    def append(self, v, fmt='B'):
        self.extend(struct.pack(fmt, v))

class NextwaveEngineComm():
    """ Class to manage:
          - Structures needed for realtime engine (boxes/refs, computed centroids, etc.)
          - Communication with the realtime engine (comm. over shared memory)
          - Computation of Zernikes, matrices for SVD, etc.
    """
    def __init__(self,parent):
        self.parent = parent
        self.ui = parent.ui

    def init(self):
        self.layout=extract_memory.get_header_format('memory_layout.h')
        self.layout_boxes=extract_memory.get_header_format('layout_boxes.h')
        self.fields=self.layout[1] # TODO: fix

        # Could be math in the defines for sizes, use eval
        MEM_LEN=int( eval(self.layout[2]['SHMEM_HEADER_SIZE'] ) )
        MEM_LEN_DATA=int(eval(self.layout[2]['SHMEM_BUFFER_SIZE'] ) )
        MEM_LEN_BOXES=self.layout_boxes[0]
        if WINDOWS:
            # TODO: Get these all from the .h defines
            self.shmem_hdr=mmap.mmap(-1,MEM_LEN,"NW_SRC0_HDR")
            self.shmem_data=mmap.mmap(-1,MEM_LEN_DATA,"NW_SRC0_BUFFER")
            self.shmem_boxes=mmap.mmap(-1,MEM_LEN_BOXES,"NW_BUFFER_BOXES")

            #from multiprocessing import shared_memory
            #self.shmem_hdr = shared_memory.SharedMemory(name="NW_SRC0_HDR" ).buf
            #self.shmem_data = shared_memory.SharedMemory(name="NW_SRC0_BUFFER" ).buf
            #self.shmem_boxes = shared_memory.SharedMemory(name="NW_BUFFER2").buf
        else:
            fd1=os.open('/dev/shm/NW_SRC0_HDR', os.O_RDWR)
            self.shmem_hdr=mmap.mmap(fd1, MEM_LEN)
            fd2=os.open('/dev/shm/NW_SRC0_BUFFER', os.O_RDWR)
            self.shmem_data=mmap.mmap(fd2,MEM_LEN_DATA)
            fd3=os.open('/dev/shm/NW_BUFFER_BOXES', os.O_RDWR)
            self.shmem_boxes=mmap.mmap(fd3,MEM_LEN_BOXES)

    def write_params(self, overrides=None):
        if overrides:
            bytez =np.array([self.parent.num_boxes], dtype="uint16").tobytes() 
            fields = self.layout_boxes[1]
            self.shmem_boxes.seek(fields['num_boxes']['bytenum_current'])
            self.shmem_boxes.write(bytez)
            self.shmem_boxes.flush()
            #print( self.parent.num_boxes)

        bytez =np.array([self.parent.ccd_pixel, self.parent.box_um, self.parent.pupil_radius_mm], dtype='double').tobytes() 
        fields = self.layout_boxes[1]
        self.shmem_boxes.seek(fields['pixel_um']['bytenum_current'])
        self.shmem_boxes.write(bytez)
        self.shmem_boxes.flush()

    def read_mode(self):
        self.shmem_hdr.seek(2) #TODO: get address
        buf=self.shmem_hdr.read(1)
        mode= struct.unpack('B',buf)[0]
        return mode

    def rcv_searchboxes(self,shmem_boxes, layout, box_x, box_y, layout_boxes):
        fields=layout[1]

        adr=fields['box_x']['bytenum_current']
        shmem_boxes.seek(adr)
        box_buf=shmem_boxes.read(NUM_BOXES*4) # TODO: ALL WRONG FOR DOUBLES
        box_x = [struct.unpack('f',box_buf[n*4:n*4+4]) for n in np.arange(NUM_BOXES)]
        #box_x =np.frombuffer(box_buf, dtype='uint8', count=NUM_BOXES )
        #print(box_x[0], box_x[1], box_x[2], box_x[3])

        adr=fields['box_y']['bytenum_current']
        shmem_boxes.seek(adr)
        box_buf=shmem_boxes.read(NUM_BOXES*4)
        box_y = [struct.unpack('f',box_buf[n*4:n*4+4]) for n in np.arange(NUM_BOXES)]
        #box_x =np.frombuffer(box_buf, dtype='uint8', count=NUM_BOXES )

        return box_x,box_y

    def send_searchboxes(self,box_x, box_y):

        shmem_boxes = self.shmem_boxes
        layout_boxes = self.layout_boxes
        defs=layout_boxes[2]
        fields=layout_boxes[1]

        num_boxes=box_x.shape[0]
        #box_size_pixel=box_size_pixel

        buf = ByteStream()
        for item in box_x:
            buf.append(item, 'd')
        shmem_boxes.seek(fields['box_x']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        buf = ByteStream()
        for item in box_y:
            buf.append(item, 'd')
        shmem_boxes.seek(fields['box_y']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        buf = ByteStream()
        for item in self.parent.ref_x:
            buf.append(item, 'd')
        shmem_boxes.seek(fields['reference_x']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        buf = ByteStream()
        for item in self.parent.ref_y:
            buf.append(item, 'd')
        shmem_boxes.seek(fields['reference_y']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        try: # TODO: Better to have an "if ready"
            if True:
                buf = ByteStream()
                for item in self.parent.influence.T.flatten():
                    buf.append(item, 'd')
                shmem_boxes.seek(fields['influence']['bytenum_current'])
                shmem_boxes.write(buf)
                shmem_boxes.flush()

            buf = ByteStream()
            for item in self.parent.influence_inv.T.flatten():
                buf.append(item, 'd')
            shmem_boxes.seek(fields['influence_inv']['bytenum_current'])
            shmem_boxes.write(buf)
            shmem_boxes.flush()
        except AttributeError:
            pass # Probably missing influence function. Hasn't been calc'ed yet.

        buf = ByteStream()
        for item in self.parent.omits:
            buf.append(item)
        shmem_boxes.seek(fields['centroid_omit']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()
        
        # Write header last, so the engine knows when we are ready
        buf = ByteStream()
        buf.append(1)
        buf.append(0)
        buf.append(num_boxes, 'H') # unsigned short
        buf.append(self.parent.ccd_pixel,'d')
        buf.append(self.parent.box_um, 'd')
        buf.append(self.parent.pupil_radius_pixel*self.parent.ccd_pixel, 'd')
        buf.append(self.parent.focal*1000.0, 'd')

        try: #TODO: Better to have a "if Ready"
            buf.append(self.parent.nTerms, 'H')
            buf.append(self.parent.nActuators, 'H')
        except AttributeError:
            pass # Ok if these aren't ready yet

        shmem_boxes.seek(0)
        shmem_boxes.write(buf)
        shmem_boxes.flush()

    def receive_image(self):
        if self.parent.ui.mode_offline or self.parent.ui.offline_only:
            # Assume it's already been written directly into image_bytes by offline processes
            try:
                return self.parent.image_bytes
            except AttributeError:
                self.parent.image_bytes=np.random.randint(0, 255, size=(1000,1000) )
                return self.parent.image_bytes

        # TODO: Wait until it's safe (unlocked)

        #TODO. Could use this method to read everything into memory. Probably more efficient:
        #self.shmem_hdr.seek(0)
        #mem_header=self.shmem_hdr.read(MEM_LEN)

        # This divider needs to match that in the engine code
        self.parent.fps0=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',0, False)/100.0
        self.parent.fps1=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',1, False)/100.0
        self.parent.fps2=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',2, False)/100.0

        self.parent.height=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'dimensions',0, False)
        self.parent.width=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'dimensions',1, False)
        self.height=self.parent.height
        self.width=self.parent.width

        self.parent.total_frames=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'total_frames',0, False)

        nwhich_buffer=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'current_frame',0, False)

        self.shmem_data.seek(self.width*self.height*nwhich_buffer)
        im_buf=self.shmem_data.read(self.width*self.height)
        bytez =np.frombuffer(im_buf, dtype='uint8', count=self.width*self.height )
        bytes2=np.reshape(bytez,( self.height,self.width)).copy()

        if len(bytes2)>0:
               bytesf = bytes2 / np.max(bytes2)
        else:
            bytes2 = np.zeros( (10,10));
            bytesf = np.zeros( (10,10));

        if False: #self.chkFollow.isChecked():
            box_x,box_y=rcv_searchboxes(self.shmem_boxes, self.layout_boxes, 0, 0, 0 )
            self.box_x = np.array(box_x)
            self.box_y = np.array(box_y)

        self.parent.image_bytes = bytes2

        return self.parent.image_bytes

    def write_mirrors(self,data):
        bytez =np.array(data, dtype="double").tobytes() 
        fields=self.layout_boxes[1] # TODO: fix
        self.shmem_boxes.seek(fields['mirror_voltages']['bytenum_current'])
        self.shmem_boxes.write(bytez)
        self.shmem_boxes.flush()

    def write_mirrors_offsets(self,data):
        bytez =np.array(data, dtype="double").tobytes() 
        fields=self.layout_boxes[1] # TODO: fix
        self.shmem_boxes.seek(fields['mirror_voltages_offsets']['bytenum_current'])
        self.shmem_boxes.write(bytez)
        self.shmem_boxes.flush()

    def zero_do(self):
        self.write_mirrors( np.zeros(97) ) # TODO: # of mirrors
        self.set_mode( 
            self.read_mode() | 0x20 )  # MODE_FORCE_AO_START TODO. Possible race condition.

    def flat_do(self):
        self.write_mirrors( self.mirror_state_flat )
        self.set_mode( 
            self.read_mode() | 0x20 )  # MODE_FORCE_AO_START TODO. Possible race condition.
        
    def flat_save(self):
        self.mirror_state_flat = np.copy(self.mirror_voltages)
    
    def zero_log_index(self):
        buf = ByteStream()
        fields=self.layout[1]
        buf.append(0)
        buf.append(0)
        buf.append(0)
        buf.append(0)
        self.shmem_hdr.seek(fields['log_index']['bytenum_current'])
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()  
        
    def do_snap(self, mode):
        buf = ByteStream()
        buf.append(mode) # TODO: MODE_CENTROIDING
        self.shmem_hdr.seek(2) #TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()    

    def receive_centroids(self):
        SIZEOF_DOUBLE=8
        fields=self.layout_boxes[1]
        num_boxes=self.parent.num_boxes
        self.shmem_boxes.seek(fields['centroid_x']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*SIZEOF_DOUBLE)
        self.parent.centroids_x=struct.unpack_from(''.join((['d']*num_boxes)), buf)

        self.shmem_boxes.seek(fields['centroid_y']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*SIZEOF_DOUBLE)
        self.parent.centroids_y=struct.unpack_from(''.join((['d']*num_boxes)), buf)

        self.shmem_boxes.seek(fields['delta_x']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*SIZEOF_DOUBLE)
        self.parent.delta_x=struct.unpack_from(''.join((['d']*num_boxes)), buf)

        self.shmem_boxes.seek(fields['delta_y']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*SIZEOF_DOUBLE)
        self.parent.delta_y=struct.unpack_from(''.join((['d']*num_boxes)), buf)

        self.shmem_boxes.seek(fields['mirror_voltages']['bytenum_current'])
        buf=self.shmem_boxes.read(self.parent.nActuators*SIZEOF_DOUBLE)
        self.mirror_voltages=np.array( struct.unpack_from(''.join((['d']*self.parent.nActuators)), buf) )

        # Debugging
        self.shmem_boxes.seek(fields['box_x_normalized']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*SIZEOF_DOUBLE)
        self.slopes_debug=np.array( struct.unpack_from(''.join((['d']*num_boxes)), buf) )

        DEBUGGING=False
        if DEBUGGING:
            print (num_boxes, np.min(self.centroids_x), np.max(self.centroids_x)  )
            for n in np.arange(num_boxes):
                if np.isnan(self.centroids_x[n]):
                    print( n, end=' ')
                    print (num_boxes, self.centroids_x[100], self.centroids_y[100]  )

    def set_mode(self, val):
        if not self.ui.offline_only:
            buf = ByteStream()
            buf.append(val) # TODO: MODE_CENTROIDING
            self.shmem_hdr.seek(2) #TODO: get address
            self.shmem_hdr.write(buf)
            self.shmem_hdr.flush()

    def set_nframes(self, val):
        #val= np.array( self.ui.edit_num_runs.text(), dtype='uint64' )
        #buf.append(val.tobytes())
        self.shmem_hdr.seek(self.fields['frames_left']['bytenum_current'])
        self.shmem_hdr.write(val.tobytes())
        self.shmem_hdr.flush()

    def write_image(self,dims,bytez):
        if self.ui.offline_only:
            #self.ui.image_pixels = bytez
            self.parent.image_bytes = bytez
        else:
            self.shmem_hdr.seek(self.fields['dimensions']['bytenum_current']) #TODO: nicer
            self.shmem_hdr.write(dims)
            self.shmem_hdr.flush()
            for nbuf in np.arange(4):
                self.shmem_data.seek(nbuf*2048*2048) # TODO
                self.shmem_data.write(bytez)
                self.shmem_data.flush()
