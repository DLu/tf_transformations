"""
This library is a reimplementation of the tf/transformations.py library.

https://github.com/ros/geometry/blob/noetic-devel/tf/src/tf/transformations.py

Original author: Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>
Laboratory for Fluorescence Dynamics, University of California, Irvine

Makes use of https://matthew-brett.github.io/transforms3d/ which is also
a reimplementation of the Gohlke's work, but this maintains the API.
"""

# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2009, The Regents of the University of California
# Copyright (c) 2021 PickNik Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math

import numpy

import transforms3d

TRANSLATION_IDENTITY = [0.0, 0.0, 0.0]
ROTATION_IDENTITY = numpy.identity(3, dtype=numpy.float64)
ZOOM_IDENTITY = [1.0, 1.0, 1.0]
SHEAR_IDENTITY = TRANSLATION_IDENTITY


def identity_matrix():
    """
    Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> numpy.allclose(I, numpy.dot(I, I))
    True
    >>> numpy.sum(I), numpy.trace(I)
    (4.0, 4.0)
    >>> numpy.allclose(I, numpy.identity(4, dtype=numpy.float64))
    True

    """
    return transforms3d.affines.compose(TRANSLATION_IDENTITY,
                                        ROTATION_IDENTITY,
                                        ZOOM_IDENTITY)


def translation_matrix(direction):
    """
    Return matrix to translate by direction vector.

    >>> v = numpy.random.random(3) - 0.5
    >>> numpy.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    return transforms3d.affines.compose(direction,
                                        ROTATION_IDENTITY,
                                        ZOOM_IDENTITY)


def translation_from_matrix(matrix):
    """
    Return translation vector from translation matrix.

    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> numpy.allclose(v0, v1)
    True

    """
    return transforms3d.affines.decompose(matrix)[0]


def reflection_matrix(point, normal):
    """
    Return matrix to mirror at plane defined by point and normal vector.

    >>> v0 = numpy.random.random(4) - 0.5
    >>> v0[3] = 1.0
    >>> v1 = numpy.random.random(3) - 0.5
    >>> R = reflection_matrix(v0, v1)
    >>> numpy.allclose(2., numpy.trace(R))
    True
    >>> numpy.allclose(v0, numpy.dot(R, v0))
    True
    >>> v2 = v0.copy()
    >>> v2[:3] += v1
    >>> v3 = v0.copy()
    >>> v2[:3] -= v1
    >>> numpy.allclose(v2, numpy.dot(R, v3))
    True

    """
    # Note: point[3] is ignored
    return transforms3d.reflections.rfnorm2aff(normal, point[:3])


def reflection_from_matrix(matrix):
    """
    Return mirror plane point and normal vector from reflection matrix.

    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = numpy.random.random(3) - 0.5
    >>> M0 = reflection_matrix(v0, v1)
    >>> point, normal = reflection_from_matrix(M0)
    >>> M1 = reflection_matrix(point, normal)
    >>> is_same_transform(M0, M1)
    True

    """
    normal, point = transforms3d.reflections.aff2rfnorm(matrix)
    return point, normal


def rotation_matrix(angle, direction, point=None):
    """
    Return matrix to rotate about axis defined by point and direction.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
    ...                                                direc, point)))
    True

    """
    return transforms3d.axangles.axangle2aff(direction, angle, point=point)


def rotation_from_matrix(matrix):
    """
    Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    direction, angle, point = transforms3d.axangles.aff2axangle(matrix)
    return angle, direction, point


def scale_matrix(factor, origin=None, direction=None):
    """
    Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    >>> v = (numpy.random.rand(4, 5) - 0.5) * 20.0
    >>> v[3] = 1.0
    >>> S = scale_matrix(-1.234)
    >>> numpy.allclose(numpy.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = random.random() * 10 - 5
    >>> origin = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)

    """
    return transforms3d.zooms.zfdir2aff(factor,
                                        direction=direction,
                                        origin=origin)


def scale_from_matrix(matrix):
    """
    Return scaling factor, origin and direction from scaling matrix.

    >>> factor = random.random() * 10 - 5
    >>> origin = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> S0 = scale_matrix(factor, origin)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True
    >>> S0 = scale_matrix(factor, origin, direct)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True

    """
    factor, direction, origin = transforms3d.zooms.aff2zfdir(matrix)
    return factor, origin, direction


def projection_matrix(point, normal, direction=None,
                      perspective=None, pseudo=False):
    """
    Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection direction, or none of both.

    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).

    >>> P = projection_matrix((0, 0, 0), (1, 0, 0))
    >>> numpy.allclose(P[1:, 1:], numpy.identity(4)[1:, 1:])
    True
    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> persp = numpy.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> P1 = projection_matrix(point, normal, direction=direct)
    >>> P2 = projection_matrix(point, normal, perspective=persp)
    >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> is_same_transform(P2, numpy.dot(P0, P3))
    True
    >>> P = projection_matrix((3, 0, 0), (1, 1, 0), (1, 0, 0))
    >>> v0 = (numpy.random.rand(4, 5) - 0.5) * 20.0
    >>> v0[3] = 1.0
    >>> v1 = numpy.dot(P, v0)
    >>> numpy.allclose(v1[1], v0[1])
    True
    >>> numpy.allclose(v1[0], 3.0-v1[1])
    True

    """
    M = numpy.identity(4)
    point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
    normal = unit_vector(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = numpy.array(perspective[:3], dtype=numpy.float64,
                                  copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = numpy.dot(perspective-point, normal)
        M[:3, :3] -= numpy.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            M[:3, :3] -= numpy.outer(normal, normal)
            M[:3, 3] = numpy.dot(point, normal) * (perspective+normal)
        else:
            M[:3, 3] = numpy.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = numpy.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = numpy.array(direction[:3], dtype=numpy.float64, copy=False)
        scale = numpy.dot(direction, normal)
        M[:3, :3] -= numpy.outer(direction, normal) / scale
        M[:3, 3] = direction * (numpy.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= numpy.outer(normal, normal)
        M[:3, 3] = numpy.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix, pseudo=False):
    """
    Return projection plane and perspective point from projection matrix.

    Return values are same as arguments for projection_matrix function:
    point, normal, direction, perspective, and pseudo.

    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> persp = numpy.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, direct)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    >>> result = projection_from_matrix(P0, pseudo=False)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> result = projection_from_matrix(P0, pseudo=True)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
    if not pseudo and len(i):
        # point: any eigenvector corresponding to eigenvalue 1
        point = numpy.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        # direction: unit eigenvector corresponding to eigenvalue 0
        l, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(l)) < 1e-8)[0]
        if not len(i):
            raise ValueError('no eigenvector corresponding to eigenvalue 0')
        direction = numpy.real(V[:, i[0]]).squeeze()
        direction /= vector_norm(direction)
        # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
        l, V = numpy.linalg.eig(M33.T)
        i = numpy.where(abs(numpy.real(l)) < 1e-8)[0]
        if len(i):
            # parallel projection
            normal = numpy.real(V[:, i[0]]).squeeze()
            normal /= vector_norm(normal)
            return point, normal, direction, None, False
        else:
            # orthogonal projection, where normal equals direction vector
            return point, direction, None, None, False
    else:
        # perspective projection
        i = numpy.where(abs(numpy.real(l)) > 1e-8)[0]
        if not len(i):
            raise ValueError(
                'no eigenvector not corresponding to eigenvalue 0')
        point = numpy.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        normal = - M[3, :3]
        perspective = M[:3, 3] / numpy.dot(point[:3], normal)
        if pseudo:
            perspective -= normal
        return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """
    Return matrix to obtain normalized device coordinates from frustum.

    The frustum bounds are axis-aligned along x (left, right),
    y (bottom, top) and z (near, far).

    Normalized device coordinates are in range [-1, 1] if coordinates are
    inside the frustum.

    If perspective is True the frustum is a truncated pyramid with the
    perspective point at origin and direction along z axis, otherwise an
    orthographic canonical view volume (a box).

    Homogeneous coordinates transformed by the perspective clip matrix
    need to be dehomogenized (divided by w coordinate).

    >>> frustum = numpy.random.rand(6)
    >>> frustum[1] += frustum[0]
    >>> frustum[3] += frustum[2]
    >>> frustum[5] += frustum[4]
    >>> M = clip_matrix(*frustum, perspective=False)
    >>> numpy.dot(M, [frustum[0], frustum[2], frustum[4], 1.0])
    array([-1., -1., -1.,  1.])
    >>> numpy.dot(M, [frustum[1], frustum[3], frustum[5], 1.0])
    array([ 1.,  1.,  1.,  1.])
    >>> M = clip_matrix(*frustum, perspective=True)
    >>> v = numpy.dot(M, [frustum[0], frustum[2], frustum[4], 1.0])
    >>> v / v[3]
    array([-1., -1., -1.,  1.])
    >>> v = numpy.dot(M, [frustum[1], frustum[3], frustum[4], 1.0])
    >>> v / v[3]
    array([ 1.,  1., -1.,  1.])

    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError('invalid frustrum')
    if perspective:
        if near <= _EPS:
            raise ValueError('invalid frustum: near <= 0')
        t = 2.0 * near
        M = ((-t/(right-left), 0.0, (right+left)/(right-left), 0.0),
             (0.0, -t/(top-bottom), (top+bottom)/(top-bottom), 0.0),
             (0.0, 0.0, -(far+near)/(far-near), t*far/(far-near)),
             (0.0, 0.0, -1.0, 0.0))
    else:
        M = ((2.0/(right-left), 0.0, 0.0, (right+left)/(left-right)),
             (0.0, 2.0/(top-bottom), 0.0, (top+bottom)/(bottom-top)),
             (0.0, 0.0, 2.0/(far-near), (far+near)/(near-far)),
             (0.0, 0.0, 0.0, 1.0))
    return numpy.array(M, dtype=numpy.float64)


def shear_matrix(angle, direction, point, normal):
    """
    Return matrix to shear by angle along direction vector on shear plane.

    The shear plane is defined by a point and normal vector. The direction
    vector must be orthogonal to the plane's normal vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the direction vector and its extent is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.cross(direct, numpy.random.random(3))
    >>> S = shear_matrix(angle, direct, point, normal)
    >>> numpy.allclose(1.0, numpy.linalg.det(S))
    True

    """
    return transforms3d.shears.sadn2aff(angle, direction, normal, point)


def shear_from_matrix(matrix):
    """
    Return shear angle, direction and plane from shear matrix.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.cross(direct, numpy.random.random(3))
    >>> S0 = shear_matrix(angle, direct, point, normal)
    >>> angle, direct, point, normal = shear_from_matrix(S0)
    >>> S1 = shear_matrix(angle, direct, point, normal)
    >>> is_same_transform(S0, S1)
    True

    """
    angle, direction, normal, point = transforms3d.shears.aff2sadn(matrix)
    return angle, direction, point, normal


def decompose_matrix(matrix):
    """
    Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix((1, 2, 3))
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> numpy.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> numpy.allclose(R0, R1)
    True

    """
    T, R, Z, S = transforms3d.affines.decompose(matrix)
    angles = euler_from_matrix(R)
    # TODO: Fill in the perspective partition of matrix
    return Z, S, angles, T, None


def compose_matrix(scale=None, shear=None, angles=None, translate=None,
                   perspective=None):
    """
    Return transformation matrix from sequence of transformations.

    This is the inverse of the decompose_matrix function.

    Sequence of transformations:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    >>> scale = numpy.random.random(3) - 0.5
    >>> shear = numpy.random.random(3) - 0.5
    >>> angles = (numpy.random.random(3) - 0.5) * (2*math.pi)
    >>> trans = numpy.random.random(3) - 0.5
    >>> persp = numpy.random.random(4) - 0.5
    >>> M0 = compose_matrix(scale, shear, angles, trans, persp)
    >>> result = decompose_matrix(M0)
    >>> M1 = compose_matrix(*result)
    >>> is_same_transform(M0, M1)
    True

    """
    T = translate if translate is not None else TRANSLATION_IDENTITY
    if angles is not None:
        R = transforms3d.euler.euler2mat(*angles)
    else:
        R = ROTATION_IDENTITY
    Z = scale if scale is not None else ZOOM_IDENTITY
    S = shear if shear is not None else SHEAR_IDENTITY
    M = transforms3d.affines.compose(T, R, Z, S)

    # TODO: Include perspective in composition
    return M


def orthogonalization_matrix(lengths, angles):
    """
    Return orthogonalization matrix for crystallographic cell coordinates.

    Angles are expected in degrees.

    The de-orthogonalization matrix is the inverse.

    >>> O = orthogonalization_matrix((10., 10., 10.), (90., 90., 90.))
    >>> numpy.allclose(O[:3, :3], numpy.identity(3, float) * 10)
    True
    >>> O = orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    >>> numpy.allclose(numpy.sum(O), 43.063229)
    True

    """
    a, b, c = lengths
    angles = numpy.radians(angles)
    sina, sinb, _ = numpy.sin(angles)
    cosa, cosb, cosg = numpy.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return numpy.array((
        (a*sinb*math.sqrt(1.0-co*co),  0.0,    0.0, 0.0),
        (-a*sinb*co,                    b*sina, 0.0, 0.0),
        (a*cosb,                       b*cosa, c,   0.0),
        (0.0,                          0.0,    0.0, 1.0)),
        dtype=numpy.float64)


def superimposition_matrix(v0, v1, scaling=False, usesvd=True):
    """
    Return matrix to transform given vector set into second vector set.

    v0 and v1 are shape (3, *) or (4, *) arrays of at least 3 vectors.

    If usesvd is True, the weighted sum of squared deviations (RMSD) is
    minimized according to the algorithm by W. Kabsch [8]. Otherwise the
    quaternion based algorithm by B. Horn [9] is used (slower when using
    this Python implementation).

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = numpy.random.rand(3, 10)
    >>> M = superimposition_matrix(v0, v0)
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> R = random_rotation_matrix(numpy.random.random(3))
    >>> v0 = ((1,0,0), (0,1,0), (0,0,1), (1,1,1))
    >>> v1 = numpy.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> v0 = (numpy.random.rand(4, 100) - 0.5) * 20.0
    >>> v0[3] = 1.0
    >>> v1 = numpy.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> S = scale_matrix(random.random())
    >>> T = translation_matrix(numpy.random.random(3)-0.5)
    >>> M = concatenate_matrices(T, R, S)
    >>> v1 = numpy.dot(M, v0)
    >>> v0[:3] += numpy.random.normal(0.0, 1e-9, 300).reshape(3, -1)
    >>> M = superimposition_matrix(v0, v1, scaling=True)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> M = superimposition_matrix(v0, v1, scaling=True, usesvd=False)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> v = numpy.empty((4, 100, 3), dtype=numpy.float64)
    >>> v[:, :, 0] = v0
    >>> M = superimposition_matrix(v0, v1, scaling=True, usesvd=False)
    >>> numpy.allclose(v1, numpy.dot(M, v[:, :, 0]))
    True

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:3]

    if v0.shape != v1.shape or v0.shape[1] < 3:
        raise ValueError('Vector sets are of wrong shape or type.')

    # move centroids to origin
    t0 = numpy.mean(v0, axis=1)
    t1 = numpy.mean(v1, axis=1)
    v0 = v0 - t0.reshape(3, 1)
    v1 = v1 - t1.reshape(3, 1)

    if usesvd:
        # Singular Value Decomposition of covariance matrix
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= numpy.outer(u[:, 2], vh[2, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = numpy.identity(4)
        M[:3, :3] = R
    else:
        # compute symmetric matrix N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = ((xx+yy+zz, yz-zy,    zx-xz,    xy-yx),
             (yz-zy,    xx-yy-zz, xy+yx,    zx+xz),
             (zx-xz,    xy+yx,   -xx+yy-zz, yz+zy),
             (xy-yx,    zx+xz,    yz+zy,   -xx-yy+zz))
        # quaternion: eigenvector corresponding to most positive eigenvalue
        l, V = numpy.linalg.eig(N)
        q = V[:, numpy.argmax(l)]
        q /= vector_norm(q)  # unit quaternion
        q = numpy.roll(q, -1)  # move w component to end
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    # scale: ratio of rms deviations from centroid
    if scaling:
        v0 *= v0
        v1 *= v1
        M[:3, :3] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # translation
    M[:3, 3] = t1
    T = numpy.identity(4)
    T[:3, 3] = -t0
    M = numpy.dot(M, T)
    return M


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """
    Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """
    rotation_matrix = transforms3d.euler.euler2mat(ai, aj, ak, axes=axes)
    return transforms3d.affines.compose(TRANSLATION_IDENTITY,
                                        rotation_matrix,
                                        ZOOM_IDENTITY)


def euler_from_matrix(matrix, axes='sxyz'):
    """
    Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    return transforms3d.euler.mat2euler(matrix, axes=axes)


def euler_from_quaternion(quaternion, axes='sxyz'):
    """
    Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def _reorder_input_quaternion(quaternion):
    """Reorder quaternion to have w term first."""
    x, y, z, w = quaternion
    return w, x, y, z


def _reorder_output_quaternion(quaternion):
    """Reorder quaternion to have w term last."""
    w, x, y, z = quaternion
    return x, y, z, w


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """
    Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    return _reorder_output_quaternion(
        transforms3d.euler.euler2quat(ai, aj, ak, axes=axes)
    )


def quaternion_about_axis(angle, axis):
    """
    Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, (1, 0, 0))
    >>> numpy.allclose(q, [0.06146124, 0, 0, 0.99810947])
    True

    """
    return _reorder_output_quaternion(
        transforms3d.quaternions.axangle2quat(axis, angle)
    )


def quaternion_matrix(quaternion):
    """
    Return 4x4 homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    rotation_matrix = transforms3d.quaternions.quat2mat(
        _reorder_input_quaternion(quaternion)
    )
    return transforms3d.affines.compose(TRANSLATION_IDENTITY,
                                        rotation_matrix,
                                        ZOOM_IDENTITY)


def quaternion_from_matrix(matrix):
    """
    Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    rotation_matrix = transforms3d.affines.decompose(matrix)[1]
    return _reorder_output_quaternion(
        transforms3d.quaternions.mat2quat(rotation_matrix)
    )


def quaternion_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions.

    >>> q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> numpy.allclose(q, [-44, -14, 48, 28])
    True

    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    w2, x2, y2, z2 = transforms3d.quaternions.qmult([w1, x1, y1, z1],
                                                    [w0, x0, y0, z0])
    return x2, y2, z2, w2


def quaternion_conjugate(quaternion):
    """
    Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    """
    return numpy.array((-quaternion[0], -quaternion[1],
                        -quaternion[2], quaternion[3]), dtype=numpy.float64)


def quaternion_inverse(quaternion):
    """
    Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> numpy.allclose(quaternion_multiply(q0, q1), [0, 0, 0, 1])
    True

    """
    return quaternion_conjugate(quaternion) / numpy.dot(quaternion, quaternion)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.

    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0.0)
    >>> numpy.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1.0, 1)
    >>> numpy.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(numpy.dot(q0, q))
    >>> numpy.allclose(2.0, math.acos(numpy.dot(q0, q1)) / angle) or
        numpy.allclose(2.0, math.acos(-numpy.dot(q0, q1)) / angle)
    True

    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = numpy.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def random_quaternion(rand=None):
    """
    Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = random_quaternion()
    >>> numpy.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quaternion(numpy.random.random(3))
    >>> q.shape
    (4,)

    """
    if rand is None:
        rand = numpy.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = numpy.sqrt(1.0 - rand[0])
    r2 = numpy.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return numpy.array((numpy.sin(t1)*r1,
                        numpy.cos(t1)*r1,
                        numpy.sin(t2)*r2,
                        numpy.cos(t2)*r2), dtype=numpy.float64)


def random_rotation_matrix(rand=None):
    """
    Return uniform random rotation matrix.

    rnd: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix()
    >>> numpy.allclose(numpy.dot(R.T, R), numpy.identity(4))
    True

    """
    return quaternion_matrix(random_quaternion(rand))


class Arcball(object):
    """
    Virtual Trackball Control.

    >>> ball = Arcball()
    >>> ball = Arcball(initial=numpy.identity(4))
    >>> ball.place([320, 320], 320)
    >>> ball.down([500, 250])
    >>> ball.drag([475, 275])
    >>> R = ball.matrix()
    >>> numpy.allclose(numpy.sum(R), 3.90583455)
    True
    >>> ball = Arcball(initial=[0, 0, 0, 1])
    >>> ball.place([320, 320], 320)
    >>> ball.setaxes([1,1,0], [-1, 1, 0])
    >>> ball.setconstrain(True)
    >>> ball.down([400, 200])
    >>> ball.drag([200, 400])
    >>> R = ball.matrix()
    >>> numpy.allclose(numpy.sum(R), 0.2055924)
    True
    >>> ball.next()

    """

    def __init__(self, initial=None):
        """
        Initialize virtual trackball control.

        initial : quaternion or rotation matrix

        """
        self._axis = None
        self._axes = None
        self._radius = 1.0
        self._center = [0.0, 0.0]
        self._vdown = numpy.array([0, 0, 1], dtype=numpy.float64)
        self._constrain = False

        if initial is None:
            self._qdown = numpy.array([0, 0, 0, 1], dtype=numpy.float64)
        else:
            initial = numpy.array(initial, dtype=numpy.float64)
            if initial.shape == (4, 4):
                self._qdown = quaternion_from_matrix(initial)
            elif initial.shape == (4, ):
                initial /= vector_norm(initial)
                self._qdown = initial
            else:
                raise ValueError('initial not a quaternion or matrix.')

        self._qnow = self._qpre = self._qdown

    def place(self, center, radius):
        """
        Place Arcball, e.g. when window size changes.

        center : sequence[2]
            Window coordinates of trackball center.
        radius : float
            Radius of trackball in window coordinates.

        """
        self._radius = float(radius)
        self._center[0] = center[0]
        self._center[1] = center[1]

    def setaxes(self, *axes):
        """Set axes to constrain rotations."""
        if axes is None:
            self._axes = None
        else:
            self._axes = [unit_vector(axis) for axis in axes]

    def setconstrain(self, constrain):
        """Set state of constrain to axis mode."""
        self._constrain = constrain is True

    def getconstrain(self):
        """Return state of constrain to axis mode."""
        return self._constrain

    def down(self, point):
        """Set initial cursor window coordinates and pick constrain-axis."""
        self._vdown = arcball_map_to_sphere(point, self._center, self._radius)
        self._qdown = self._qpre = self._qnow

        if self._constrain and self._axes is not None:
            self._axis = arcball_nearest_axis(self._vdown, self._axes)
            self._vdown = arcball_constrain_to_axis(self._vdown, self._axis)
        else:
            self._axis = None

    def drag(self, point):
        """Update current cursor window coordinates."""
        vnow = arcball_map_to_sphere(point, self._center, self._radius)

        if self._axis is not None:
            vnow = arcball_constrain_to_axis(vnow, self._axis)

        self._qpre = self._qnow

        t = numpy.cross(self._vdown, vnow)
        if numpy.dot(t, t) < _EPS:
            self._qnow = self._qdown
        else:
            q = [t[0], t[1], t[2], numpy.dot(self._vdown, vnow)]
            self._qnow = quaternion_multiply(q, self._qdown)

    def next(self, acceleration=0.0):  # noqa: A003 (for backwards compat.)
        """Continue rotation in direction of last drag."""
        q = quaternion_slerp(self._qpre, self._qnow, 2.0+acceleration, False)
        self._qpre, self._qnow = self._qnow, q

    def matrix(self):
        """Return homogeneous rotation matrix."""
        return quaternion_matrix(self._qnow)


def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v = numpy.array(((point[0] - center[0]) / radius,
                     (center[1] - point[1]) / radius,
                     0.0), dtype=numpy.float64)
    n = v[0]*v[0] + v[1]*v[1]
    if n > 1.0:
        v /= math.sqrt(n)  # position outside of sphere
    else:
        v[2] = math.sqrt(1.0 - n)
    return v


def arcball_constrain_to_axis(point, axis):
    """Return sphere point perpendicular to axis."""
    v = numpy.array(point, dtype=numpy.float64, copy=True)
    a = numpy.array(axis, dtype=numpy.float64, copy=True)
    v -= a * numpy.dot(a, v)  # on plane
    n = vector_norm(v)
    if n > _EPS:
        if v[2] < 0.0:
            v *= -1.0
        v /= n
        return v
    if a[2] == 1.0:
        return numpy.array([1, 0, 0], dtype=numpy.float64)
    return unit_vector([-a[1], a[0], 0])


def arcball_nearest_axis(point, axes):
    """Return axis, which arc is nearest to point."""
    point = numpy.array(point, dtype=numpy.float64, copy=False)
    nearest = None
    mx = -1.0
    for axis in axes:
        t = numpy.dot(arcball_constrain_to_axis(point, axis), point)
        if t > mx:
            nearest = axis
            mx = t
    return nearest


# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = {v: k for k, v in _AXES2TUPLE.items()}


def vector_norm(data, axis=None, out=None):
    """
    Return length, i.e. eucledian norm, of ndarray along axis.

    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3), dtype=numpy.float64)
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1.0])
    1.0

    """
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)


def unit_vector(data, axis=None, out=None):
    """
    Return ndarray normalized by length, i.e. eucledian norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float64)
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1.0]))
    [1.0]

    """
    if out is None:
        data = numpy.array(data, dtype=numpy.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(numpy.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = numpy.array(data, copy=False)
        data = out
    length = numpy.atleast_1d(numpy.sum(data*data, axis))
    numpy.sqrt(length, length)
    if axis is not None:
        length = numpy.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def random_vector(size):
    """
    Return array of random doubles in the half-open interval [0.0, 1.0).

    >>> v = random_vector(10000)
    >>> numpy.all(v >= 0.0) and numpy.all(v < 1.0)
    True
    >>> v0 = random_vector(10)
    >>> v1 = random_vector(10)
    >>> numpy.any(v0 == v1)
    False

    """
    return numpy.random.random(size)


def inverse_matrix(matrix):
    """
    Return inverse of square transformation matrix.

    >>> M0 = random_rotation_matrix()
    >>> M1 = inverse_matrix(M0.T)
    >>> numpy.allclose(M1, numpy.linalg.inv(M0.T))
    True
    >>> for size in range(1, 7):
    ...     M0 = numpy.random.rand(size, size)
    ...     M1 = inverse_matrix(M0)
    ...     if not numpy.allclose(M1, numpy.linalg.inv(M0)): print size

    """
    return numpy.linalg.inv(matrix)


def concatenate_matrices(*matrices):
    """
    Return concatenation of series of transformation matrices.

    >>> M = numpy.random.rand(16).reshape((4, 4)) - 0.5
    >>> numpy.allclose(M, concatenate_matrices(M))
    True
    >>> numpy.allclose(numpy.dot(M, M.T), concatenate_matrices(M, M.T))
    True

    """
    M = numpy.identity(4)
    for i in matrices:
        M = numpy.dot(M, i)
    return M


def is_same_transform(matrix0, matrix1):
    """
    Return True if two matrices perform same transformation.

    >>> is_same_transform(numpy.identity(4), numpy.identity(4))
    True
    >>> is_same_transform(numpy.identity(4), random_rotation_matrix())
    False

    """
    matrix0 = numpy.array(matrix0, dtype=numpy.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = numpy.array(matrix1, dtype=numpy.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return numpy.allclose(matrix0, matrix1)
