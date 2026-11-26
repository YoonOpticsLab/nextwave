import numpy as np
from scipy.optimize import minimize

class circle_fitter():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        
    def circle_err(self,p):
        cx,cy,rad=p
        X=self.x
        Y=self.y
        self.losses = (X-cx)**2 +(Y-cy)**2 - rad**2
        self.loss = np.sum( self.losses**2 )
        return self.loss

    def solve(self):
        guess = np.array ( [np.mean( self.x), np.mean(self.y), 0] )
        guess[2] = np.sqrt( np.mean( (guess[0]-self.x)**2 + (guess[1]-self.y)**2 ) )
        self.guess = guess
        maxR = np.sqrt( ((self.x.max() - self.x.min())/2)**2 + ((self.y.max() - self.y.min())/2) **2 )
        self.bounds = [[self.x.min(),self.x.max()] ,[self.y.min(),self.y.max()],[0, maxR ] ]
        opt1=minimize( self.circle_err, guess, method='Nelder-Mead', bounds=self.bounds );
        best = opt1['x']
        self.params = best

#im_t = gaussian_filter(im1,GAUSS_SD)
#val = filters.threshold_otsu(im_t)
#print( "Otsu threshold=%d"%val )
#im_t = im_subd.copy()
#im_t[im_t<val] = 0
        
			
			
# First attempt tried to use:
        # https://www.algothingy.com/blogs/least_squares_circle_fit.html        
# but there were several errors in the post, so	now using straightforward fit to circumference line
		
            # self.params = [best[0]/2, best[1]/2, np.sqrt(best[2] + (best[0]**2+best[1]**2)/4.0 ) ]
			            # Using circle equation, solving for params
            #A=2*cx
            #B=2*cy
            #C=cx**2+cy**2-rad**2
            #loss = 0.5 * np.sum( (X**2+Y**2-A*X-B*Y-C) ** 2 )
