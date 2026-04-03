import numpy as np
import matplotlib.pyplot as plt

def find_indices1(boxes_diameter, aperture):
    #max_cells = np.sqrt( half_m1    )
    max_cells = boxes_diameter
    cells_x = np.arange(-(max_cells-1)/2,(max_cells-1)/2+1) # +1 to include +max_boxes number
    cells_y = np.arange(-(max_cells-1)/2,(max_cells-1)/2+1)
    
    # Determine outer edge of each box using corners away from the center:
    # 0.5*sign: positive adds 0.5, negative substracts 0.5
    XX,YY = np.meshgrid(cells_x, cells_y )
    #aperture = self.pupil_radius_pixel / self.box_spacing_pixel * 1.0
    RR = np.sqrt( (XX+0.5*np.sign(XX))**2 + (YY+0.5*np.sign(YY))**2 )
    valid_cells = np.where(RR.flatten()<aperture)[0]
    # For each valid index, need to make both the X and Y component
    #        valid_indices = np.concatenate( (valid_boxes*2, valid_boxes*2+1) )
    valid_indices = valid_cells
    return valid_indices    
 
#return stacked indices (e.g. for influence matrix)
def find_indices2(half_m1, aperture):
    max_cells = np.sqrt( half_m1 / 2.0 )
    cells_x = np.arange(-(max_cells-1)/2,(max_cells-1)/2+1) # +1 to include +max_boxes number
    cells_y = np.arange(-(max_cells-1)/2,(max_cells-1)/2+1)
    
    # Determine outer edge of each box using corners away from the center:
    # 0.5*sign: positive adds 0.5, negative substracts 0.5
    XX,YY = np.meshgrid(cells_x, cells_y )
    #aperture = self.pupil_radius_pixel / self.box_spacing_pixel * 1.0
    RR = np.sqrt( (XX+0.5*np.sign(XX))**2 + (YY+0.5*np.sign(YY))**2 )
    valid_cells = np.where(RR.flatten()<aperture)[0]
    # For each valid index, need to make both the X and Y component
#        valid_indices = np.concatenate( (valid_boxes*2, valid_boxes*2+1) )
    valid_indices = np.vstack( (valid_cells*2, valid_cells*2+1) ).T.flatten()
    return valid_indices

def idxs_sparse(N):
    #max_cells = np.sqrt( half_m1    )
    r = np.ceil( np.sqrt(N/3.14) )
    diam = int( (r-1)*2+1 )
    max_cells = diam
    aperture=np.sqrt(N/3.14)+0.5
    cells_x = np.arange(-(max_cells-1)/2,(max_cells-1)/2+1) # +1 to include +max_boxes number
    cells_y = np.arange(-(max_cells-1)/2,(max_cells-1)/2+1)
    
    # Determine outer edge of each box using corners away from the center:
    # 0.5*sign: positive adds 0.5, negative substracts 0.5
    XX,YY = np.meshgrid(cells_x, cells_y )
    #aperture = self.pupil_radius_pixel / self.box_spacing_pixel * 1.0
    RR = np.sqrt( (XX+0.5*np.sign(XX))**2 + (YY+0.5*np.sign(YY))**2 )
    valid_cells = np.where(RR.flatten()<=aperture)[0]
    # For each valid index, need to make both the X and Y component
    #        valid_indices = np.concatenate( (valid_boxes*2, valid_boxes*2+1) )
    #valid_indices = valid_cells
    return XX.flatten()[valid_cells]+r-1,YY.flatten()[valid_cells]+r-1
    
    
    
    
def boxes_nrows(rows, x, y, cx, cy, lenslet_size, pixels_um,do_plot=False):
    XX = (x-cx) / (lenslet_size/pixels_um)
    YY = (y-cy) / (lenslet_size/pixels_um)
    RR = np.sqrt( XX**2 + YY**2 )
    
    boxes_in = RR <=rows
    boxes_out= RR > rows
    
    idxs=np.arange(len(x))
    
    
    if do_plot:
        plt.axis('equal')
        plt.plot( XX[boxes_in], YY[boxes_in], 'go' )
        plt.plot( XX[boxes_out], YY[boxes_out], 'rx' )
        
    return idxs[boxes_in], idxs[boxes_out], RR