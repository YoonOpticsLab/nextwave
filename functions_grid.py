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
    
def build_dense(arrx,arry):
    rx_sorted=np.unique( arrx )
    ry_sorted=np.unique( arry )
    
    r=np.ceil( np.sqrt( len(arrx)/3.14 ) )
    diam = int( (r-1)*2+1 )
    print( diam, diam*diam, len(arrx) )
    
    to_dense_map=np.zeros( (diam,diam), dtype='int')-1
    for nref,rx1 in enumerate(arrx):
        nx=np.where( (rx_sorted == arrx[nref] ) )
        ny=np.where( (ry_sorted == arry[nref] ) )
    
        to_dense_map[ny,nx] = int(nref)

    return to_dense_map
    
class sparse_grid():
    def __init__(self,n_act):
        self.n_act=n_act
        self.xact,self.yact=idxs_sparse(n_act)
        self.to_dense_map=build_dense(self.xact,self.yact)
        return
        
    def idx_to_yx(self,idx):
        matches=np.where( (idx == self.to_dense_map ))
        return matches[0][0],matches[1][0]
    def yx_to_idx(self,y,x):
        return self.to_dense_map[y,x]
        
    def build_laplacian_sparse(self):
        N = self.n_act
        L = np.zeros((N, N))
        for idx in range(self.n_act):
            y,x=self.idx_to_yx(idx)
            neighbors = []
            if x > 0: neighbors.append(self.yx_to_idx(y,x-1))
            if x < self.to_dense_map.shape[1]-1: neighbors.append(self.yx_to_idx(y,x+1))
            if y > 0: neighbors.append(self.yx_to_idx(y-1,x))
            if y < self.to_dense_map.shape[0]-1: neighbors.append(self.yx_to_idx(y+1,x))
            valids=0
            for n in neighbors:
                if n==-1: continue # Not valid idx on this grid
                L[idx, n] = -1
                valids += 1
            L[idx,idx] = valids
        return L    