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

"""Tests for tf_transformations."""
import math
import random

import numpy

from tf_transformations import _AXES2TUPLE, clip_matrix, euler_from_quaternion
from tf_transformations import compose_matrix, concatenate_matrices
from tf_transformations import decompose_matrix, euler_from_matrix
from tf_transformations import euler_matrix, identity_matrix, inverse_matrix
from tf_transformations import is_same_transform, orthogonalization_matrix
from tf_transformations import projection_from_matrix, projection_matrix
from tf_transformations import quaternion_about_axis, quaternion_conjugate
from tf_transformations import quaternion_from_euler, quaternion_from_matrix
from tf_transformations import quaternion_inverse, quaternion_matrix
from tf_transformations import quaternion_multiply, quaternion_slerp
from tf_transformations import random_quaternion, random_rotation_matrix
from tf_transformations import random_vector, rotation_matrix
from tf_transformations import reflection_from_matrix, reflection_matrix
from tf_transformations import rotation_from_matrix, scale_from_matrix
from tf_transformations import scale_matrix, shear_from_matrix, shear_matrix
from tf_transformations import superimposition_matrix, translation_matrix
from tf_transformations import translation_from_matrix, unit_vector
from tf_transformations import vector_norm


def test_standard():
    alpha, beta, gamma = 0.123, -1.234, 2.345
    origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    Rx = rotation_matrix(alpha, xaxis)
    Ry = rotation_matrix(beta, yaxis)
    Rz = rotation_matrix(gamma, zaxis)
    R = concatenate_matrices(Rx, Ry, Rz)
    euler = euler_from_matrix(R, 'rxyz')
    assert numpy.allclose([alpha, beta, gamma], euler)

    Re = euler_matrix(alpha, beta, gamma, 'rxyz')
    assert is_same_transform(R, Re)

    al, be, ga = euler_from_matrix(Re, 'rxyz')
    assert is_same_transform(Re, euler_matrix(al, be, ga, 'rxyz'))

    qx = quaternion_about_axis(alpha, xaxis)
    qy = quaternion_about_axis(beta, yaxis)
    qz = quaternion_about_axis(gamma, zaxis)
    q = quaternion_multiply(qx, qy)
    q = quaternion_multiply(q, qz)
    Rq = quaternion_matrix(q)
    assert is_same_transform(R, Rq)

    S = scale_matrix(1.23, origin)
    T = translation_matrix((1, 2, 3))
    Z = shear_matrix(beta, xaxis, origin, zaxis)
    R = random_rotation_matrix(numpy.random.rand(3))
    M = concatenate_matrices(T, R, Z, S)
    scale, shear, angles, trans, persp = decompose_matrix(M)
    assert numpy.allclose(scale, 1.23)

    assert numpy.allclose(trans, (1, 2, 3))

    assert numpy.allclose(shear, (0, math.tan(beta), 0))

    assert is_same_transform(R, euler_matrix(axes='sxyz', *angles))

    M1 = compose_matrix(scale, shear, angles, trans, persp)
    assert is_same_transform(M, M1)


def test_identity_matrix():
    m1 = identity_matrix()
    assert numpy.allclose(m1, numpy.dot(m1, m1))
    assert numpy.sum(m1) == 4.0
    assert numpy.trace(m1) == 4.0
    assert numpy.allclose(m1, numpy.identity(4, dtype=numpy.float64))


def test_translation_matrix():
    v = numpy.random.random(3) - 0.5
    assert numpy.allclose(v, translation_matrix(v)[:3, 3])


def test_translation_from_matrix():
    v0 = numpy.random.random(3) - 0.5
    v1 = translation_from_matrix(translation_matrix(v0))
    assert numpy.allclose(v0, v1)


def test_reflection_matrix():
    v0 = numpy.random.random(4) - 0.5
    v0[3] = 1.0
    v1 = numpy.random.random(3) - 0.5
    R = reflection_matrix(v0, v1)
    assert numpy.allclose(2., numpy.trace(R))
    assert numpy.allclose(v0, numpy.dot(R, v0))
    v2 = v0.copy()
    v2[:3] += v1
    v3 = v0.copy()
    v2[:3] -= v1
    assert numpy.allclose(v2, numpy.dot(R, v3))


def test_reflection_from_matrix():
    v0 = numpy.random.random(3) - 0.5
    v1 = numpy.random.random(3) - 0.5
    M0 = reflection_matrix(v0, v1)
    point, normal = reflection_from_matrix(M0)
    M1 = reflection_matrix(point, normal)
    assert is_same_transform(M0, M1)


def test_rotation_matrix():
    angle = (random.random() - 0.5) * (2*math.pi)
    direc = numpy.random.random(3) - 0.5
    point = numpy.random.random(3) - 0.5
    R0 = rotation_matrix(angle, direc, point)
    R1 = rotation_matrix(angle-2*math.pi, direc, point)
    assert is_same_transform(R0, R1)
    R0 = rotation_matrix(angle, direc, point)
    R1 = rotation_matrix(-angle, -direc, point)
    assert is_same_transform(R0, R1)
    m1 = numpy.identity(4, numpy.float64)
    assert numpy.allclose(m1, rotation_matrix(math.pi*2, direc))
    m2 = numpy.trace(rotation_matrix(math.pi/2, direc, point))
    assert numpy.allclose(2., m2)


def test_rotation_from_matrix():
    angle = (random.random() - 0.5) * (2*math.pi)
    direc = numpy.random.random(3) - 0.5
    point = numpy.random.random(3) - 0.5
    R0 = rotation_matrix(angle, direc, point)
    angle, direc, point = rotation_from_matrix(R0)
    R1 = rotation_matrix(angle, direc, point)
    assert is_same_transform(R0, R1)


def test_scale_matrix():
    v = (numpy.random.rand(4, 5) - 0.5) * 20.0
    v[3] = 1.0
    S = scale_matrix(-1.234)
    assert numpy.allclose(numpy.dot(S, v)[:3], -1.234*v[:3])
    factor = random.random() * 10 - 5
    origin = numpy.random.random(3) - 0.5
    direct = numpy.random.random(3) - 0.5
    S = scale_matrix(factor, origin)
    S = scale_matrix(factor, origin, direct)


def test_scale_from_matrix():
    factor = random.random() * 10 - 5
    origin = numpy.random.random(3) - 0.5
    direct = numpy.random.random(3) - 0.5
    S0 = scale_matrix(factor, origin)
    factor, origin, direction = scale_from_matrix(S0)
    S1 = scale_matrix(factor, origin, direction)
    assert is_same_transform(S0, S1)
    S0 = scale_matrix(factor, origin, direct)
    factor, origin, direction = scale_from_matrix(S0)
    S1 = scale_matrix(factor, origin, direction)
    assert is_same_transform(S0, S1)


def test_projection_matrix():
    P = projection_matrix((0, 0, 0), (1, 0, 0))
    assert numpy.allclose(P[1:, 1:], numpy.identity(4)[1:, 1:])
    point = numpy.random.random(3) - 0.5
    normal = numpy.random.random(3) - 0.5
    # direct = numpy.random.random(3) - 0.5
    persp = numpy.random.random(3) - 0.5
    P0 = projection_matrix(point, normal)
    # P1 = projection_matrix(point, normal, direction=direct)
    P2 = projection_matrix(point, normal, perspective=persp)
    P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    assert is_same_transform(P2, numpy.dot(P0, P3))
    P = projection_matrix((3, 0, 0), (1, 1, 0), (1, 0, 0))
    v0 = (numpy.random.rand(4, 5) - 0.5) * 20.0
    v0[3] = 1.0
    v1 = numpy.dot(P, v0)
    assert numpy.allclose(v1[1], v0[1])
    assert numpy.allclose(v1[0], 3.0-v1[1])


def test_projection_from_matrix():
    point = numpy.random.random(3) - 0.5
    normal = numpy.random.random(3) - 0.5
    direct = numpy.random.random(3) - 0.5
    persp = numpy.random.random(3) - 0.5
    P0 = projection_matrix(point, normal)
    result = projection_from_matrix(P0)
    P1 = projection_matrix(*result)
    assert is_same_transform(P0, P1)
    P0 = projection_matrix(point, normal, direct)
    result = projection_from_matrix(P0)
    P1 = projection_matrix(*result)
    assert is_same_transform(P0, P1)
    P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    result = projection_from_matrix(P0, pseudo=False)
    P1 = projection_matrix(*result)
    assert is_same_transform(P0, P1)
    P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    result = projection_from_matrix(P0, pseudo=True)
    P1 = projection_matrix(*result)
    assert is_same_transform(P0, P1)


def test_clip_matrix():
    frustum = numpy.random.rand(6)
    frustum[1] += frustum[0]
    frustum[3] += frustum[2]
    frustum[5] += frustum[4]
    M = clip_matrix(*frustum, perspective=False)
    assert numpy.allclose(
        numpy.dot(M, [frustum[0], frustum[2], frustum[4], 1.0]),
        [-1., -1., -1.,  1.])
    assert numpy.allclose(
        numpy.dot(M, [frustum[1], frustum[3], frustum[5], 1.0]),
        [1.,  1.,  1.,  1.])
    M = clip_matrix(*frustum, perspective=True)
    v = numpy.dot(M, [frustum[0], frustum[2], frustum[4], 1.0])
    assert numpy.allclose(v / v[3], [-1., -1., -1.,  1.])
    v = numpy.dot(M, [frustum[1], frustum[3], frustum[4], 1.0])
    assert numpy.allclose(v / v[3], [1.,  1., -1.,  1.])


def test_shear_matrix():
    angle = (random.random() - 0.5) * 4*math.pi
    direct = numpy.random.random(3) - 0.5
    point = numpy.random.random(3) - 0.5
    normal = numpy.cross(direct, numpy.random.random(3))
    S = shear_matrix(angle, direct, point, normal)
    assert numpy.allclose(1.0, numpy.linalg.det(S))


def test_shear_from_matrix():
    angle = (random.random() - 0.5) * 4*math.pi
    direct = numpy.random.random(3) - 0.5
    point = numpy.random.random(3) - 0.5
    normal = numpy.cross(direct, numpy.random.random(3))
    S0 = shear_matrix(angle, direct, point, normal)
    angle, direct, point, normal = shear_from_matrix(S0)
    S1 = shear_matrix(angle, direct, point, normal)
    assert is_same_transform(S0, S1)


def test_decompose_matrix():
    T0 = translation_matrix((1, 2, 3))
    scale, shear, angles, trans, persp = decompose_matrix(T0)
    T1 = translation_matrix(trans)
    assert numpy.allclose(T0, T1)
    S = scale_matrix(0.123)
    scale, shear, angles, trans, persp = decompose_matrix(S)
    assert scale[0] == 0.123
    R0 = euler_matrix(1, 2, 3)
    scale, shear, angles, trans, persp = decompose_matrix(R0)
    R1 = euler_matrix(*angles)
    assert numpy.allclose(R0, R1)


def test_compose_matrix():
    scale = numpy.random.random(3) - 0.5
    shear = numpy.random.random(3) - 0.5
    angles = (numpy.random.random(3) - 0.5) * (2*math.pi)
    trans = numpy.random.random(3) - 0.5
    persp = numpy.random.random(4) - 0.5
    M0 = compose_matrix(scale, shear, angles, trans, persp)
    result = decompose_matrix(M0)
    M1 = compose_matrix(*result)
    assert is_same_transform(M0, M1)


def test_orthogonalization_matrix():
    om = orthogonalization_matrix((10., 10., 10.), (90., 90., 90.))
    assert numpy.allclose(om[:3, :3], numpy.identity(3, float) * 10)
    om = orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    assert numpy.allclose(numpy.sum(om), 43.063229)


def test_superimposition_matrix():
    v0 = numpy.random.rand(3, 10)
    M = superimposition_matrix(v0, v0)
    assert numpy.allclose(M, numpy.identity(4))
    R = random_rotation_matrix(numpy.random.random(3))
    v0 = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
    v1 = numpy.dot(R, v0)
    M = superimposition_matrix(v0, v1)
    assert numpy.allclose(v1, numpy.dot(M, v0))
    v0 = (numpy.random.rand(4, 100) - 0.5) * 20.0
    v0[3] = 1.0
    v1 = numpy.dot(R, v0)
    M = superimposition_matrix(v0, v1)
    assert numpy.allclose(v1, numpy.dot(M, v0))
    S = scale_matrix(random.random())
    T = translation_matrix(numpy.random.random(3)-0.5)
    M = concatenate_matrices(T, R, S)
    v1 = numpy.dot(M, v0)
    v0[:3] += numpy.random.normal(0.0, 1e-9, 300).reshape(3, -1)
    M = superimposition_matrix(v0, v1, scaling=True)
    assert numpy.allclose(v1, numpy.dot(M, v0))
    M = superimposition_matrix(v0, v1, scaling=True, usesvd=False)
    assert numpy.allclose(v1, numpy.dot(M, v0))
    v = numpy.empty((4, 100, 3), dtype=numpy.float64)
    v[:, :, 0] = v0
    M = superimposition_matrix(v0, v1, scaling=True, usesvd=False)
    assert numpy.allclose(v1, numpy.dot(M, v[:, :, 0]))


def test_euler_matrix():
    R = euler_matrix(1, 2, 3, 'syxz')
    assert numpy.allclose(numpy.sum(R[0]), -1.34786452)
    R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    assert numpy.allclose(numpy.sum(R[0]), -0.383436184)


def test_euler_from_matrix():
    R0 = euler_matrix(1, 2, 3, 'syxz')
    al, be, ga = euler_from_matrix(R0, 'syxz')
    R1 = euler_matrix(al, be, ga, 'syxz')
    assert numpy.allclose(R0, R1)
    angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    for axes in _AXES2TUPLE.keys():
        R0 = euler_matrix(axes=axes, *angles)
        R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
        assert numpy.allclose(R0, R1)


def test_euler_from_quaternion():
    angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    assert numpy.allclose(angles, [0.123, 0, 0])


def test_quaternion_from_euler():
    q = quaternion_from_euler(1, 2, 3, 'ryxz')
    assert numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])


def test_quaternion_about_axis():
    q = quaternion_about_axis(0.123, (1, 0, 0))
    assert numpy.allclose(q, [0.06146124, 0, 0, 0.99810947])


def test_quaternion_matrix():
    R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    assert numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))


def test_quaternion_from_matrix():
    R = rotation_matrix(0.123, (1, 2, 3))
    q = quaternion_from_matrix(R)
    assert numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])


def test_quaternion_multiply():
    q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    assert numpy.allclose(q, [-44, -14, 48, 28])


def test_quaternion_conjugate():
    q0 = random_quaternion()
    q1 = quaternion_conjugate(q0)
    assert q1[3] == q0[3] and all(q1[:3] == -q0[:3])


def test_quaternion_inverse():
    q0 = random_quaternion()
    q1 = quaternion_inverse(q0)
    assert numpy.allclose(quaternion_multiply(q0, q1), [0, 0, 0, 1])


def test_quaternion_slerp():
    q0 = random_quaternion()
    q1 = random_quaternion()
    q = quaternion_slerp(q0, q1, 0.0)
    assert numpy.allclose(q, q0)
    q = quaternion_slerp(q0, q1, 1.0, 1)
    assert numpy.allclose(q, q1)
    q = quaternion_slerp(q0, q1, 0.5)
    angle = math.acos(numpy.dot(q0, q))
    assert numpy.allclose(2.0, math.acos(numpy.dot(q0, q1)) / angle) or \
        numpy.allclose(2.0, math.acos(-numpy.dot(q0, q1)) / angle)


def test_random_quaternion():
    q = random_quaternion()
    assert numpy.allclose(1.0, vector_norm(q))
    q = random_quaternion(numpy.random.random(3))
    assert q.shape == (4,)


def test_random_rotation_matrix():
    R = random_rotation_matrix()
    assert numpy.allclose(numpy.dot(R.T, R), numpy.identity(4))


def test_vector_norm():
    v = numpy.random.random(3)
    n = vector_norm(v)
    assert numpy.allclose(n, numpy.linalg.norm(v))
    v = numpy.random.rand(6, 5, 3)
    n = vector_norm(v, axis=-1)
    assert numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    n = vector_norm(v, axis=1)
    assert numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    v = numpy.random.rand(5, 4, 3)
    n = numpy.empty((5, 3), dtype=numpy.float64)
    vector_norm(v, axis=1, out=n)
    assert numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    assert vector_norm([]) == 0.0
    assert vector_norm([1.0]) == 1.0


def test_unit_vector():
    v0 = numpy.random.random(3)
    v1 = unit_vector(v0)
    assert numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    v0 = numpy.random.rand(5, 4, 3)
    v1 = unit_vector(v0, axis=-1)
    v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    assert numpy.allclose(v1, v2)
    v1 = unit_vector(v0, axis=1)
    v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    assert numpy.allclose(v1, v2)
    v1 = numpy.empty((5, 4, 3), dtype=numpy.float64)
    unit_vector(v0, axis=1, out=v1)
    assert numpy.allclose(v1, v2)
    assert list(unit_vector([])) == []
    assert list(unit_vector([1.0])) == [1.0]


def test_random_vector():
    v = random_vector(10000)
    assert numpy.all(v >= 0.0) and numpy.all(v < 1.0)
    v0 = random_vector(10)
    v1 = random_vector(10)
    assert not numpy.any(v0 == v1)


def test_inverse_matrix():
    M0 = random_rotation_matrix()
    M1 = inverse_matrix(M0.T)
    assert numpy.allclose(M1, numpy.linalg.inv(M0.T))
    for size in range(1, 7):
        M0 = numpy.random.rand(size, size)
        M1 = inverse_matrix(M0)
        assert numpy.allclose(M1, numpy.linalg.inv(M0))


def test_concatenate_matrices():
    M = numpy.random.rand(16).reshape((4, 4)) - 0.5
    assert numpy.allclose(M, concatenate_matrices(M))
    assert numpy.allclose(numpy.dot(M, M.T), concatenate_matrices(M, M.T))


def test_is_same_transform():
    assert is_same_transform(numpy.identity(4), numpy.identity(4))
    assert not is_same_transform(numpy.identity(4), random_rotation_matrix())
