"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2
import similar
from sympy import symbols, Eq, solve



def camera_matrixp1(K1):
    I = np.identity(3)
    c = [[0],
        [0],
        [0]]
    I_0 = np.hstack((I, c))

    p1 = K1 @ I_0

    return p1

def essential_matrix(F, K1, K2):
    e = K2.T @ F @ K1

    return e 

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):

    T = np.array([[1/M, 0, 0], 
         [0, 1/M, 0], 
         [0, 0, 1]])

    points1 = np.zeros(pts1.shape, dtype = float)
    points2 = np.zeros(pts1.shape, dtype = float)

    #normalizing the points
    for i in range(len(pts1)):
        points1[i][0] = float(pts1[i][0])/M
        points1[i][1] = float(pts1[i][1])/M

        points2[i][0] = float(pts2[i][0])/M
        points2[i][1] = float(pts2[i][1])/M
    
    F, _ = cv2.findFundamentalMat(points1, points2)


    U, s, V = np.linalg.svd(F)  #or can directly use helper function
    x = min(range(len(s)), key=lambda i: abs(s[i]))
    s[x] = 0
    s_ = np.diag(s)
    F_ = U @s_ @ V

    F_rect = hlp.refineF(F_, points1, points2)

    F_final = T.transpose() @ F_rect @ T

    return F_final


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""

def epipolar_correspondences(im1, im2, F, pts1):

    #setting a threshold for point lying on line
    thresh = 0.005
    copts = []
    for pt in pts1:     
        x = [[pt[0]],
             [pt[1]], 
             [1]]

        #equation of line
        epline = F @ x
        matches = []

        qw = 100 #half width of query region square

        for xc in range(max(0, pt[0] -qw), min(pt[0] + qw,im2.shape[1])):
            for yc in range(max(0, pt[1] -qw), min(pt[1] + qw,im2.shape[0])):

                p = [xc, yc, 1]
                m = p @ epline

                if(abs(m) < thresh):    
                    matches.append((xc, yc))
        
        copt = similar.most_similar(im1, im2, pt, matches)
        copts.append(copt)

    return copts



"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    e = K2.T @ F @ K1

    return e 


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    xcoods = []
    ycoods = []
    zcoods = []
    for k in range(len(pts2)):
        x, y, z = symbols('x,y,z')

        A = []
        b = []

        s = []
        for i in range(3):
            s.append(P1[2][i]*pts1[k][0] - P1[0][i])
        
        A.append(s)
        b.append(P1[2][3]*pts1[k][0] - P1[0][3])

        s = []
        for i in range(3):
            s.append(P1[2][i]*pts1[k][1] - P1[0][i])
        
        A.append(s)
        b.append(P1[2][3]*pts1[k][1] - P1[0][3])

        s = []
        for i in range(3):
            s.append(P2[2][i]*pts2[k][1] - P2[0][i])
        
        A.append(s)
        b.append(P2[2][3]*pts2[k][1] - P2[0][3])

        s = []
        for i in range(3):
            s.append(P2[2][i]*pts2[k][1] - P2[0][i])
        
        A.append(s)
        b.append(P2[2][3]*pts2[k][1] - P2[0][3])

        Sol = np.linalg.solve(np.array(A[:3]), np.array(list(-x for x in b[:3])))

        xcoods.append(Sol[0])
        ycoods.append(Sol[1])
        zcoods.append(Sol[2])
    
    
     
    # Creating figure
    #fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(xcoods, ycoods, zcoods, color = "green")
    plt.title("simple 3D scatter plot")
    
    # show plot
    plt.show()


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
