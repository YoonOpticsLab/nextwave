import sys
import os
import mmap
import time

import numpy as np
from numpy.linalg import svd
import tkinter as tk
from multiprocessing import shared_memory
import extract_memory

# TODO: Move these into defauls/configs/editable box
reps=3
positions=[-0.2,0.2]

# SHMEM Stuff. TODO: Put in standalone python class
nptypes={'double':np.float64, 'uint16_t':np.uint16,'uint8_t':np.uint8}
def shmem_item(shmem1,layout,varname):
    item_info=layout[1][varname]
    data=np.frombuffer(shmem1, dtype=nptypes[item_info['type']],
                       count=item_info['num_elements'], offset=item_info['bytenum_current'])
    if len(data)==1:
        data=data[0]
    return data

def shmem_write(shmem1,layout,varname,data):
    item_info=layout[1][varname]
    bytez =np.array(data, dtype=nptypes[item_info['type']]).tobytes()   
    offset=item_info['bytenum_current']
    shmem1[offset : offset + len(bytez)] = bytez

class NW_shmem:
    def __init__(self,mem,layout):
        self.mem = mem
        self.layout = layout

    def item(self,name):
        return shmem_item(self.mem, self.layout, name)

    def write(self,name,data):
        return shmem_write(self.mem, self.layout, name, data)

    def get_fields(self):
        return list( self.layout[1].keys() )
        
def connect_shmem():
    global mem0, mem_boxes
    global shmem_boxes, shmem_main # These need to be global so it doesn't try to close the memory (Surprisingly NW_shmem object life doesn't help)

    layout=extract_memory.get_header_format('memory_layout.h')
    NAME='NW_SRC0_HDR'
    shmem_main = shared_memory.SharedMemory(name=NAME)

    layout_log=extract_memory.get_header_format('layout_log.h')
    NAME='NW_LOG'
    #shm_log = shared_memory.SharedMemory(name=NAME, size=1024*1024)

    layoutb=extract_memory.get_header_format('layout_boxes.h')
    shmem_boxes = shared_memory.SharedMemory(name="NW_BUFFER_BOXES")
    buf_boxes=shmem_boxes.buf
    
    mem_boxes = NW_shmem(buf_boxes,layoutb)

    mem0 = NW_shmem(shmem_main.buf,layout)

    num_boxes=mem_boxes.item('num_boxes')
    num_mirs=mem_boxes.item('nActuators')
    s=f"OK nBoxes={num_boxes}, nMirrors={num_mirs}" 
    print( s )#, centroid_x[0],  centroid_y[0] )
    label1.config( text = s )
    return


def do_cal():
    global infl_mat
    
    remove_tiptilt = check_var.get()

    num_boxes=mem_boxes.item('num_boxes')
    num_mirs=mem_boxes.item('nActuators')
    reference_x=mem_boxes.item('reference_x')[0:num_boxes]
    reference_y=mem_boxes.item('reference_y')[0:num_boxes]

    mirs=np.zeros(num_mirs)
    centroids=np.zeros( (len(positions), 2, reps, len(mirs), num_boxes ) )
    for npos,pos1 in enumerate(positions):
        for nrep in np.arange(reps):
            print(f"\n\n{pos1} {nrep}",end=" ", flush=True)
            for nmirror in np.arange(num_mirs):
                mirs *= 0
                mirs[nmirror] = pos1
                mem_boxes.write("mirror_voltages",mirs)
                time.sleep(0.1)

                displ_x = mem_boxes.item('centroid_x')[0:num_boxes].copy() - reference_x
                displ_y = mem_boxes.item('centroid_y')[0:num_boxes].copy() - reference_y

                # Remove tip/tilt
                centroids[npos,0,nrep,nmirror] =    displ_x - np.mean( displ_x )*remove_tiptilt
                centroids[npos,1,nrep,nmirror] = - (displ_y - np.mean( displ_y )*remove_tiptilt  )

                print(f"{nmirror}", end=" ", flush=True)
                
    # THIS IS THE RECAL SEQUENCE STEP #2 (Average and write)

    # Average the 3 repeats, separate the two positions/directions
    pos1=np.nanmean(centroids[0,:,:,:], axis=1 ) 
    pos2=np.nanmean(centroids[1,:,:,:], axis=1 )

    # Subtract positive from negative position
    diff = (pos1-pos2) / (positions[0]-positions[1] ) 
    diff = diff * mem_boxes.item('pixel_um') / mem_boxes.item('focal_um') # convert from pixels

    # to Visualize
    #which_mirror = 48
    #plt.plot( diff[0,which_mirror]  )
    #plt.plot( diff[1,which_mirror]  )

    # Reshape into a grid, then flatten and interleave
    x_uniq = np.sort( np.unique( reference_x ) )
    y_uniq = np.sort( np.unique( reference_y ) )
    size_big_grid = int( len(x_uniq) )
    idx_to_grid = np.zeros( (2, len(reference_x) ), dtype='int' )
    
    # TODO: Better comments. It's not obvious the how/why of this code.
    for nbox in np.arange(len(reference_x)):
        # Put y as first dim. so it's in array order
        idx_to_grid[0,nbox] = ( np.where( reference_y[nbox] == y_uniq )[0] )
        idx_to_grid[1,nbox] = ( np.where( reference_x[nbox] == x_uniq )[0] )

    def circ_array_to_grid(infx, infy):
        grid2=np.zeros( (infx.shape[0], size_big_grid, size_big_grid, 2) )
        for nbox in np.arange(infx.shape[1]):
            grid2[:,idx_to_grid[0,nbox],idx_to_grid[1,nbox], 0] = infx[:,nbox]
            grid2[:,idx_to_grid[0,nbox],idx_to_grid[1,nbox], 1] = infy[:,nbox]
        grid2 = np.reshape( grid2, (infx.shape[0], 2 * size_big_grid**2 ) )
        return grid2
    infl_mat=circ_array_to_grid( diff[0], diff[1] )
    return
    


def do_reinit():
    connect_shmem()
def do_validate():
    # Starting from the padded matrix canvas, reduce back down to valid and see if SVD succeeds. Print cond#s to terminals.
    valids=np.sum(infl_mat**2, 0 )>0
    infl = infl_mat[ :,valids  ]
    
    print( np.sum( infl_mat**2 ), np.sum(infl**2), valids, infl.shape )
    
    try:
        U, s, V = svd(infl, full_matrices=True)
        s_recip = 1/s
        condition = s[0]/s[::-1]
        print( condition )
    except:
        print( "SVD failed" )
        
    timestr = time.strftime("%Y%m%d-%H%M%S")        
    np.savetxt( f"influence-{timestr}.txt", infl_mat )
    
    return infl_mat
    
def do_replace():
    # Uses global infl_mat
    
    # TODO: Get filename from XML
    np.savetxt( "/MiniWave/MiniWaveConfiguration/InfluenceMatrix_ALPAO_7.5mm_new.txt", infl_mat )

        
# 1. Create the main window
root = tk.Tk()
root.title("Nextwave calibration")
root.geometry("400x400")

btn_init = tk.Button(root, text="Reinit", command=do_reinit)
btn_init.pack(pady=10) # Add some vertical padding

# Create a Label widget
label1 = tk.Label(
    root,
    text="Please click Init",  # The text to display
    font=("Helvetica", 16, "bold"), # Set the font and size
    bg="#f0f0f0",  # Background color
    fg="blue"  # Foreground (text) color
)

check_var = tk.BooleanVar()
check_var.set(True) # Set initial state to unchecked

# Create the Checkbutton widget
checkbox = tk.Checkbutton(
    root,
    text="Remove Tip/Tilt during calibration",
    variable=check_var,
  #  command=on_checkbox_change
)


label1.pack(pady=5)
checkbox.pack(pady=5)

btn_runcal = tk.Button(root, text="Run calibration sequence", command=do_cal)
btn_runcal.pack(pady=5) 
btn_runval = tk.Button(root, text="Validate new matrix SVD", command=do_validate)
btn_runval.pack(pady=5) 
btn_runrep = tk.Button(root, text="Replace Nextwave matrix with new", command=do_replace)
btn_runrep.pack(pady=5) 


# 3. Create a second, minimal 'Quit' button
# The 'command=root.destroy' closes the main window
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack(pady=5)

# 4. Start the application's main loop
root.mainloop()