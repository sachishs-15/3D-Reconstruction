import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2
import similar
from sympy import symbols, Eq, solve

def triangulate(P1, P2, pts1, pts2):

    # points1 = np.zeros(pts1.shape, dtype = float)
    # points2 = np.zeros(pts1.shape, dtype = float)

    # for i in range(len(pts1)):
    #     points1[i][0] = float(pts1[i][0])/M
    #     points1[i][1] = float(pts1[i][1])/M

    #     points2[i][0] = float(pts2[i][0])/M
    #     points2[i][1] = float(pts2[i][1])/M

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
    
    # lineq = []

    # for i in len(range(points1)):
    #     x1 = points1[i][0]
    #     y1 = points1[i][1]

    #     x2 = points2[i][0]
    #     y2 = points2[i][1]

    #     lineq.append([x1,*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    
    F, _ = cv2.findFundamentalMat(points1, points2)

    #print(F)

    U, s, V = np.linalg.svd(F)  #or can directly use helper function
    x = min(range(len(s)), key=lambda i: abs(s[i]))
    s[x] = 0
    s_ = np.diag(s)
    F_ = U @s_ @ V

    F_rect = hlp.refineF(F_, points1, points2)

    F_final = T.transpose() @ F_rect @ T

    return F_final

    #print(F_final)

    #hlp.displayEpipolarF(img1, img2, F_final)

            


    


def epipolar_correspondences(im1, im2, F, pts1):

    #setting a threshold for point lying on line
    thresh = 1
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







# 1. Load the two temple images and the points from data/some_corresp.npz

img1 = cv2.imread('data/im1.png')
img2 = cv2.imread('data/im2.png')

    # cv2.imshow("IMG1", img1)
    # cv2.imshow("IMG2", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

data = np.load('data/some_corresp.npz')

print(data)

c1 = data['pts1']
c2 = data['pts2']

# 2. Run eight_point to compute F

M = float(max(img1.shape[0], img1.shape[1]))
F = eight_point(c1, c2, M)

print(F)

# 3. Load points in image 1 from data/temple_coords.npz

temple_coords = np.load('data/temple_coords.npz')
pts1 = temple_coords['pts1']

# 4. Run epipolar_correspondences to get points in image 2

pts2 = epipolar_correspondences(img1, img2, F, pts1)
#hlp.epipolarMatchGUI(img1, img2, F)
for i in range(len(pts1)):
    p1 = pts1[i]
    p2 = pts2[i]

    cv2.circle(img1, (p1[0],p1[1]), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(img2, (p2[0],p2[1]), radius=3, color=(0, 255, 0), thickness=-1)

    cv2.imshow("IMG1", img1)
    cv2.imshow("IMG2", img2)

    cv2.waitKey(5000)

#4.1 Getting Essential Matrix

data = np.load('data/intrinsics.npz')
K1 = data['K1']
K2 = data['K2']
print(K1)
E = essential_matrix(F, K1, K2)

# 5. Compute the camera projection matrix P1

P1 = camera_matrixp1(K1)

# 6. Use camera2 to get 4 camera projection matrices P2

possibleext = hlp.camera2(E)

P2_choices = [] #4 possible extrinsic matrices


for i in range(4):
    P2_choices.append(K2@(possibleext[:,:,i]))

# 7. Run triangulate using the projection matrices
for p2 in P2_choices:
    triangulate(P1, p2, pts1, pts2)
    

# 8. Figure out the correct P2

# 9. Scatter plot the correct 3D points

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
