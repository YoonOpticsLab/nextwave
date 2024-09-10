from numpy import sqrt, zeros, concatenate

# Taken from GYY's MATLAB code and converted to Python
def extrapolate_zernikes(z_in,pupil_ratio):
    num_coeffs=z_in.shape[0]
    z=z_in * 0 # Will be replaced
    b=concatenate( ([0], z_in ) )
    a=pupil_ratio

    a2 = a*a;
    a2m1 = a2 - 1.0;
    a3 = a*a2;
    a4 = a2*a2;
    a5 = a2*a3;
    a6 = a3*a3;
    a7 = a3*a4;
    a8 = a4*a4;
    a9 = a4*a5;
    a10 = a5*a5;
    sqrt2 = sqrt(2.0);
    sqrt3 = sqrt(3.0);
    sqrt5 = sqrt(5.0);
    sqrt6 = sqrt(6.0);
    sqrt7 = sqrt(7.0);
    sqrt10 = sqrt(10.0);
    sqrt11 = sqrt(11.0);
    sqrt15 = sqrt(15.0);
    sqrt21 = sqrt(21.0);
    sqrt33 = sqrt(33.0);
    sqrt35 = sqrt(35.0);
    sqrt55 = sqrt(55.0);
    sqrt77 = sqrt(77.0);

    #         z[0] = 0 - sqrt3*a2m1*b[4]/a2 - sqrt5*a2m1*(-1 + 2*a2)*b[12]/a4; %+ sqrt7*a2m1*(1 - 5*a2 + 5*a4)*c(22) + 3*a2m1*(-1 + 2*a2)*(1 - 7*a2 + 7*a4)*c(37) + sqrt11*a2m1*(1 - 14*a2 + 56*a4 - 84*a6 + 42*a8)*c(56);

    z[2] = (b[2] - 2*sqrt2*a2m1*a*b[8]/a3 - sqrt3*a2m1*a*(-3 + 5*a2)*b[18]/a5)/a;# + 4*a2m1*a*(2 - 8*a2 + 7*a4)*c(30) + sqrt5*a2m1*a*(-5 + 35*a2 - 70*a4 + 42*a6)*c(46);
    z[1] = (b[1] - 2*sqrt2*a2m1*a*b[7]/a3 - sqrt3*a2m1*a*(-3 + 5*a2)*b[17]/a5)/a;# + 4*a2m1*a*(2 - 8*a2 + 7*a4)*c(29) + sqrt5*a2m1*a*(-5 + 35*a2 - 70*a4 + 42*a6)*c(47);

    z[4] = (b[4] - sqrt15*a2m1*a2*b[12]/a4)/a2;# + sqrt21*a2m1*a2*(-2 + 3*a2)*c(22) + sqrt3*a2m1*a2*(10 - 35*a2 + 28*a4)*c(37) + sqrt33*a2m1*a2*(-5 + 30*a2 - 54*a4 + 30*a6)*c(56);
    z[3] = (b[3] - sqrt15*a2m1*a2*b[11]/a4)/a2;# + sqrt21*a2m1*a2*(-2 + 3*a2)*c(23) + sqrt3*a2m1*a2*(10 - 35*a2 + 28*a4)*c(39) + sqrt33*a2m1*a2*(-5 + 30*a2 - 54*a4 + 30*a6)*c(57);
    z[5] = (b[5] - sqrt15*a2m1*a2*b[13]/a4)/a2;# + sqrt21*a2m1*a2*(-2 + 3*a2)*c(24) + sqrt3*a2m1*a2*(10 - 35*a2 + 28*a4)*c(38) + sqrt33*a2m1*a2*(-5 + 30*a2 - 54*a4 + 30*a6)*c(58);

    z[7] = (b[7] - 2*sqrt6*a2m1*a3*b[17]/a5)/a3;# + 2*sqrt2*a2m1*a3*(-5 + 7*a2)*c(29) + 2*sqrt10*a2m1*a3*(-1 + 2*a2)*(-5 + 6*a2)*c(47);
    z[8] = (b[8] - 2*sqrt6*a2m1*a3*b[18]/a5)/a3;# + 2*sqrt2*a2m1*a3*(-5 + 7*a2)*c(30) + 2*sqrt10*a2m1*a3*(-1 + 2*a2)*(-5 + 6*a2)*c(46);
    z[6] = (b[6] - 2*sqrt6*a2m1*a3*b[16]/a5)/a3;# + 2*sqrt2*a2m1*a3*(-5 + 7*a2)*c(31) + 2*sqrt10*a2m1*a3*(-1 + 2*a2)*(-5 + 6*a2)*c(49);
    z[9] = (b[9] - 2*sqrt6*a2m1*a3*b[19]/a5)/a3;# + 2*sqrt2*a2m1*a3*(-5 + 7*a2)*c(32) + 2*sqrt10*a2m1*a3*(-1 + 2*a2)*(-5 + 6*a2)*c(48);

    z[12] = (b[12] - sqrt35*a2m1*a4*b[24]/a6)/a4;# + 3*sqrt5*a2m1*a4*(-3 + 4*a2)*c(37) + sqrt55*a2m1*a4*(7 - 21*a2 + 15*a4)*c(56);
    z[13] = b[13]/a4;# + sqrt35*a2m1*a4*c(24) + 3*sqrt5*a2m1*a4*(-3 + 4*a2)*c(38) + sqrt55*a2m1*a4*(7 - 21*a2 + 15*a4)*c(58);
    z[11] = b[11]/a4;# + sqrt35*a2m1*a4*c(23) + 3*sqrt5*a2m1*a4*(-3 + 4*a2)*c(39) + sqrt55*a2m1*a4*(7 - 21*a2 + 15*a4)*c(57);
    z[14] = b[14]/a4;# + sqrt35*a2m1*a4*c(26) + 3*sqrt5*a2m1*a4*(-3 + 4*a2)*c(40) + sqrt55*a2m1*a4*(7 - 21*a2 + 15*a4)*c(60);
    z[10] = b[10]/a4;# + sqrt35*a2m1*a4*c(25) + 3*sqrt5*a2m1*a4*(-3 + 4*a2)*c(41) + sqrt55*a2m1*a4*(7 - 21*a2 + 15*a4)*c(59);

    return concatenate( (z[1:],[0] ) )
