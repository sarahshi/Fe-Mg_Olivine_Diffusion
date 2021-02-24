# Code modified from Britton et al 2016 "Tutorial: Crystal orientations and EBSD - Or which way is up?"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math as m
import matplotlib.image as image

# alpha = np.deg2rad(alpha) # convert angles into radians
# beta  = np.deg2rad(beta)
# gamma = np.deg2rad(gamma)
#
## Euler Angles in Bunge convention
# phi1=float(BRKR_phi1)*m.pi/180.0
# Phi =float(BRKR_Phi) *m.pi/180.0
# phi2=float(BRKR_phi2)*m.pi/180.0


# ------------------------------------------------------------------------------
# Rotation matrices
# ------------------------------------------------------------------------------
#  see e.g.
#  J. B. Kuipers "Quaternions and Rotation Sequences",
#  Princeton University Press, 1999


def Rx(RotAngle):
    """
    provides the X axis (e1) rotation matrix in cartesian systems,
	input "RotAngle" in radians

    meaning for a COLUMN of VECTORS: transformation matrix U for VECTORS
    this matrix rotates a set of "old" basis vectors
    $(\vec{e1},\vec{e2},\vec{e3})^T$ (column) by +RotAngle (right hand rule)
    to a new set of basis vectors $(\vec{e1}',\vec{e2}',\vec{e3}')^T$ (column)

    meaning for a COLUMN of vector COORDINATES:
    (1) (N_P_O):    coordinates of fixed vector in a "new" basis that is
                    rotated by +RotAngle (passive rotation)
    (2) (O_P_O):    active rotation of vector coordinates in the same
                    "old" basis by -RotAngle
    """
    mat = np.matrix(
        [
            [1, 0, 0],
            [0, np.cos(RotAngle), np.sin(RotAngle)],
            [0, -np.sin(RotAngle), np.cos(RotAngle)],
        ]
    )
    return mat


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
def Rz(RotAngle):
    """
    provides the Z axis (e3) rotation matrix in cartesian systems,
	input "RotAngle" in radians

    meaning for a COLUMN of VECTORS: transformation matrix U for VECTORS
    this matrix rotates a set of "old" basis vectors
    $(\vec{e1},\vec{e2},\vec{e3})^T$ (column) by +RotAngle (right hand rule)
    to a new set of basis vectors $(\vec{e1}',\vec{e2}',\vec{e3}')^T$ (column)

    meaning for a COLUMN of vector COORDINATES:
    (1) (N_P_O):    coordinates of fixed vector in a "new" basis that is
                    rotated by +RotAngle (passive rotation)
    (2) (O_P_O):    active rotation of vector coordinates in the same
                    "old" basis by -RotAngle
    """
    mat = np.matrix(
        [
            [np.cos(RotAngle), np.sin(RotAngle), 0],
            [-np.sin(RotAngle), np.cos(RotAngle), 0],
            [0, 0, 1],
        ]
    )
    return mat


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
def CalcStructureMatrix(
    a=1.0,
    b=1.0,
    c=1.0,
    alpha=np.deg2rad(90.0),
    beta=np.deg2rad(90.0),
    gamma=np.deg2rad(90.0),
):
    """
    computes the structure matrix from lattice parameters
    input angles in RADIANS
    convention: McKie&McKie, "Essentials of Crystallography", 1986
    """

    ca = m.cos(alpha)
    sa = m.sin(alpha)
    cb = m.cos(beta)
    cg = m.cos(gamma)

    ax = a * m.sqrt(1.0 + 2.0 * ca * cb * cg - (ca * ca + cb * cb + cg * cg)) / sa
    ay = a * (cg - ca * cb) / sa
    az = a * cb

    by = b * sa
    bz = b * ca

    cz = c

    StructureMat = np.matrix([[ax, 0, 0], [ay, by, 0], [az, bz, cz]])
    return StructureMat


# ------------------------------------------------------------------------------
# STRUCTURE MATRICES
# ------------------------------------------------------------------------------
# direct structure matrix A
# A gives the direct lattice basis vector coordinates (column vectors)
# in the Cartesian system
# A = CalcStructureMatrix(a,b,c,alpha,beta,gamma)
# reciprocal structure matrix A+: transpose of inverse of A
# A+ gives to reciprocal lattice basis vector coordinates (column vectors)
# in the Cartesian system
# Aplus = (A.I).T
# ------------------------------------------------------------------------------

# rotation of Sample System to Cartesian Crystal System = ORIENTATION OF GRAIN
#

# I am using Row Vectors in accoradnace  with Britton et al 2016
# it is important to remember the order of multiplcation row row vectors xA=B not Ax=b(column vectors)

# U_O = Rz(phi2)  * Rx(Phi) * Rz(phi1)


def Crystal2Sample(uvw, phi1, Phi, phi2):
    uvw = np.matrix(uvw)

    U_O = Rz(np.deg2rad(phi2)) * Rx(np.deg2rad(Phi)) * Rz(np.deg2rad(phi1))

    Structmat = CalcStructureMatrix(
        a=1,
        b=1,
        c=1,  # right now its for the cubic system. I need to determine if we need vectors in the basis of for the unit cell of its just the raw angle
        alpha=np.deg2rad(90.0),  # olivine a=4.78,b=10.25,c=6.3
        beta=np.deg2rad(90.0),
        gamma=np.deg2rad(90.0),
    )

    # Sample_xyz= uvw * Structmat.T * U_O # commented out to remove struct matrix
    Sample_xyz = uvw * U_O

    return Sample_xyz


def Sample2Crystal(xyz, phi1, Phi, phi2):
    uvw = np.matrix(xyz)

    U_O = Rz(np.deg2rad(phi2)) * Rx(np.deg2rad(Phi)) * Rz(np.deg2rad(phi1))

    Structmat = CalcStructureMatrix(
        a=1,
        b=1,
        c=1,  # right now its for the cubic system. I need to determine if we need vectors in the basis of for the unit cell of its just the raw angle
        alpha=np.deg2rad(90.0),  # olivine a=4.78,b=10.25,c=6.3
        beta=np.deg2rad(90.0),
        gamma=np.deg2rad(90.0),
    )

    Sample_xyz = (
        uvw * U_O.T
    )  # * (Structmat.T).I #commented out to remove structure matrix

    return Sample_xyz


# Dan's points in radians
phi1 = np.rad2deg(0.85967)
PHI = np.rad2deg(0.13492)
phi2 = np.rad2deg(0.14829)

# Sample coordinate rotations from positive x axis counter clockwise


def profile_angle(theta):  # Theta in degrees from the positive X axis
    # x = np.cos(np.deg2rad(theta))
    # y = np.sin(np.deg2rad(theta))
    # z = 0.0
    x = np.sin(
        np.deg2rad(theta)
    )  # this configuration has the x axis oriented to the north
    y = -np.cos(
        np.deg2rad(theta)
    )  # This has a negative sign because our SEM is oriented left to right. I may need to change this if the Euler angles dont represent the orientation corrected
    z = 0.0
    return [x, y, z]


#
# Sample2Crystal(xyz, phi1, PHI, phi2)
# Out[50]: matrix([[ 0.14178132,  0.07091429, -0.01764734]])
#
# uvw = Sample2Crystal(xyz, phi1, PHI, phi2)
#
# Crystal2Sample(uvw, phi1, PHI, phi2)
# Out[52]: matrix([[ -2.58819045e-01,   9.65925826e-01,  -1.38777878e-17]])


def Shortest_angle(uvw, axis):

    dot = np.dot(uvw, axis.T)
    vector_norm = np.linalg.norm(uvw)
    axis_norm = np.linalg.norm(axis)

    short_angle = np.arccos(dot / (axis_norm * vector_norm))

    short_angle = np.rad2deg(short_angle)

    return short_angle


# alpha=27.2675453
# def vector_direction(theta,phi1,Phi,phi2): # Flawed Method!!! Doesnt work !!!!!
#    xyz = profile_angle(theta)
#    #uvw = Sample2Crystal(xyz,phi1,Phi,phi2)
#    uvw = Crystal2Sample(xyz,phi1,Phi,phi2)  #This method doesn't work because I am rotating both the vector and Axes. I need to do one or the other
#
#    #Ac = Sample2Crystal(np.matrix([0,1,0]),phi1,Phi,phi2)
#    #Bc = Sample2Crystal(np.matrix([1,0,0]),phi1,Phi,phi2)#B and X aligned to the East
#    #Cc = Sample2Crystal(np.matrix([0,0,1]),phi1,Phi,phi2)
#    Ac = Sample2Crystal(np.matrix([1,0,0]),phi1,Phi,phi2)
#    Bc = Sample2Crystal(np.matrix([0,1,0]),phi1,Phi,phi2)# B and Y aligned to the East
#    Cc = Sample2Crystal(np.matrix([0,0,1]),phi1,Phi,phi2)
#    print "Apha :" + str(min(Shortest_angle(uvw,Ac),Shortest_angle(uvw,-Ac)))
#    print "Beta :" + str(min(Shortest_angle(uvw,Bc),Shortest_angle(uvw,-Bc)))
#    print "Gamma:" + str(min(Shortest_angle(uvw,Cc),Shortest_angle(uvw,-Cc)))


def vector_direction2(
    theta, phi1, Phi, phi2
):  # Sample Rotated using Crystal 2 Sample (Works as Expected for AMNH EBSD)
    uvw = np.matrix(profile_angle(theta))
    # uvw = Sample2Crystal(xyz,phi1,Phi,phi2)

    # Ac = Sample2Crystal(np.matrix([0,1,0]),phi1,Phi,phi2)
    # Bc = Sample2Crystal(np.matrix([1,0,0]),phi1,Phi,phi2)#B and X aligned to the East
    # Cc = Sample2Crystal(np.matrix([0,0,1]),phi1,Phi,phi2)
    # Ac = Sample2Crystal(np.matrix([1,0,0]),phi1,Phi,phi2)
    # Bc = Sample2Crystal(np.matrix([0,1,0]),phi1,Phi,phi2)# B and Y aligned to the East
    # Cc = Sample2Crystal(np.matrix([0,0,1]),phi1,Phi,phi2)
    Ac = Crystal2Sample(np.matrix([1, 0, 0]), phi1, Phi, phi2)
    Bc = Crystal2Sample(
        np.matrix([0, 1, 0]), phi1, Phi, phi2
    )  # B and Y aligned to the East
    Cc = Crystal2Sample(np.matrix([0, 0, 1]), phi1, Phi, phi2)

    print("Apha :" + str(min(Shortest_angle(uvw, Ac), Shortest_angle(uvw, -Ac))))
    print("Beta :" + str(min(Shortest_angle(uvw, Bc), Shortest_angle(uvw, -Bc))))
    print("Gamma:" + str(min(Shortest_angle(uvw, Cc), Shortest_angle(uvw, -Cc))))


# try rotating axis not vector and try using x rotation matrix


def vector_direction3(theta, phi1, Phi, phi2):  # Sample Rotated using Sample 2 Crystal
    uvw = np.matrix(profile_angle(theta))

    # Ac = Sample2Crystal(np.matrix([0,1,0]),phi1,Phi,phi2)
    # Bc = Sample2Crystal(np.matrix([1,0,0]),phi1,Phi,phi2)#B and X aligned to the East
    # Cc = Sample2Crystal(np.matrix([0,0,1]),phi1,Phi,phi2)
    Ac = Sample2Crystal(np.matrix([1, 0, 0]), phi1, Phi, phi2)
    Bc = Sample2Crystal(
        np.matrix([0, 1, 0]), phi1, Phi, phi2
    )  # B and Y aligned to the East
    Cc = Sample2Crystal(np.matrix([0, 0, 1]), phi1, Phi, phi2)
    print("Apha :" + str(min(Shortest_angle(uvw, Ac), Shortest_angle(uvw, -Ac))))
    print("Beta :" + str(min(Shortest_angle(uvw, Bc), Shortest_angle(uvw, -Bc))))
    print("Gamma:" + str(min(Shortest_angle(uvw, Cc), Shortest_angle(uvw, -Cc))))


def vector_direction4(
    theta, phi1, Phi, phi2
):  # Vector Rotated using Sample to Crystal (Works as Expected for AMNH EBSD)
    xyz = profile_angle(theta)
    uvw = Sample2Crystal(xyz, phi1, Phi, phi2)

    # Ac = Sample2Crystal(np.matrix([0,1,0]),phi1,Phi,phi2)
    # Bc = Sample2Crystal(np.matrix([1,0,0]),phi1,Phi,phi2)#B and X aligned to the East
    # Cc = Sample2Crystal(np.matrix([0,0,1]),phi1,Phi,phi2)
    Ac = np.matrix([1, 0, 0])
    Bc = np.matrix([0, 1, 0])  # B and Y aligned to the East
    Cc = np.matrix([0, 0, 1])
    print("Apha :" + str(min(Shortest_angle(uvw, Ac), Shortest_angle(uvw, -Ac))))
    print("Beta :" + str(min(Shortest_angle(uvw, Bc), Shortest_angle(uvw, -Bc))))
    print("Gamma:" + str(min(Shortest_angle(uvw, Cc), Shortest_angle(uvw, -Cc))))


def vector_direction5(theta, phi1, Phi, phi2):  # Vector Rotated using Sample to Crystal
    xyz = profile_angle(theta)
    uvw = Crystal2Sample(xyz, phi1, Phi, phi2)

    # Ac = Sample2Crystal(np.matrix([0,1,0]),phi1,Phi,phi2)
    # Bc = Sample2Crystal(np.matrix([1,0,0]),phi1,Phi,phi2)#B and X aligned to the East
    # Cc = Sample2Crystal(np.matrix([0,0,1]),phi1,Phi,phi2)
    Ac = np.matrix([1, 0, 0])
    Bc = np.matrix([0, 1, 0])  # B and Y aligned to the East
    Cc = np.matrix([0, 0, 1])
    print("Apha :" + str(min(Shortest_angle(uvw, Ac), Shortest_angle(uvw, -Ac))))
    print("Beta :" + str(min(Shortest_angle(uvw, Bc), Shortest_angle(uvw, -Bc))))
    print("Gamma:" + str(min(Shortest_angle(uvw, Cc), Shortest_angle(uvw, -Cc))))
