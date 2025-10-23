import numpy as np

def find_indices(mat1, aperture):
    max_cells = np.sqrt( mat1.shape[1] / 2.0 )
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
 