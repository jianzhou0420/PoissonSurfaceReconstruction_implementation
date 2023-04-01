import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import cg
import mcubes
import time


def fd_partial_derivative(nx, ny, nz, h, direction):
    '''

    :param nx: number of divisions on x-axis
    :param ny: number of divisions on y-axis
    :param nz: number of divisions on z-axis
    :param h: size of division
    :param direction: x or y or z
    :return: the partial_derivative of the cube on one direction

    # as we know, for a sparse matrix, gradient is obtained by (x[i,i]-x[i,i+1])/interval
    # that is use 'this pixel minus the next pixel and then divide interval'
    # the coo_matrix method take data, [row,col] as input where the i-th element in data are located at [row[i],col[i]]
    # of the total matrix so the x[i,i] has locations of row=[0,1,2,3,...],col=[0,1,2,3,...]
    # and x[i,i+1] has locations of row=[0,1,2,3,...], col=[1,2,3,4,....]
    # according to (x[i,i]-x[i,i+1])/interval, the coefficients of x[i,i] and x[i,i+1] are 1/interval(1/h)
    # and -1/interval(-1/h), respectively.


    # but in the discussion above, we didn't pay attention to the boundary conditions. If we just follow the procedure
    above, the last col's gradient will be calculated by the formula as follows(in 2D conditions)
    [(the last pixel of i-th row)-(the first pixel of i+1-th row)]/interval
    So we need to handle the boundary properly.

    # Handle the boundary condition
        There are four popular methods to take care of boundary pixels
        1. Nearest-neighbor interpolation:
        2. Zero-padding:
        3. Mirror padding:
        4. Periodic padding:

    in this method, the author choose to set the gradient of the boundary to zero, that is mirror padding
    '''

    # set grid id
    primary_grid_idx = np.arange(nx * ny * nz).reshape((nx, ny, nz))  # nx changes the fastest; nz changes the slowest.

    if direction == "x":
        # bottom is positive, top is negative
        num_staggered_grid = (nx - 1) * ny * nz
        # total number of grid, but why the length of the x-axis was minused by 1
        # oh he is handling the boundary conditions, as the index of boundary is not taken by col_idx

        # But for g[i,i], we don't need to delete any element, the front term of the equation is not affected by
        # boundary,this could be wrong, lets modify it
        col_idx = np.concatenate((primary_grid_idx[1:, ...].flatten(), primary_grid_idx[:-1, :, :].flatten()))
        # col_idx = np.concatenate((primary_grid_idx.flatten(), primary_grid_idx[:-1, :, :].flatten()))
    elif direction == "y":
        # right is positive, left is negative
        num_staggered_grid = nx * (ny - 1) * nz
        col_idx = np.concatenate((primary_grid_idx[:, 1:, :].flatten(), primary_grid_idx[:, :-1, :].flatten()))
        # col_idx = np.concatenate((primary_grid_idx.flatten(), primary_grid_idx[:, :-1, :].flatten()))
    elif direction == "z":
        # back is positive, front is negative
        num_staggered_grid = nx * ny * (nz - 1)
        col_idx = np.concatenate((primary_grid_idx[:, :, 1:].flatten(), primary_grid_idx[:, :, :-1].flatten()))
        # col_idx = np.concatenate((primary_grid_idx.flatten(), primary_grid_idx[:, :, :-1].flatten()))

    row_idx = np.arange(num_staggered_grid)
    row_idx = np.tile(row_idx, 2)

    data_term = [1 / h] * num_staggered_grid + [-1 / h] * num_staggered_grid
    D = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_staggered_grid, nx * ny * nz))

    return D


def fd_grad(nx, ny, nz, hx, hy, hz):
    '''

    :param nx: number of cubes on x-axis
    :param ny: number of cubes on y-axis
    :param nz: number of cubes on z-axis
    :param hx: size of cubes on x-axis
    :param hy: size of cubes on y-axis
    :param hz: size of cubes on z-axis
    :return:
    '''
    Dx = fd_partial_derivative(nx, ny, nz, hx, "x")
    Dy = fd_partial_derivative(nx, ny, nz, hy, "y")
    Dz = fd_partial_derivative(nx, ny, nz, hz, "z")
    return vstack((Dx, Dy, Dz))


def trilinear_interpolation_weights(nx, ny, nz, corner, P, hx, hy, hz, direction=None):
    '''

    :param nx: number of cubes on x-axis
    :param ny: number of cubes on y-axis
    :param nz: number of cubes on z-axis
    :param corner: the coordinate of the most bottem-left-front corner point
    :param P: samples
    :param hx: size of cubes on x-axis
    :param hy: size of cubes on y-axis
    :param hz: size of cubes on z-axis
    :param direction: direction
    :return: after inserting values to the samples
    '''
    if direction == "x":  # center of the corner cube
        corner[0] += 0.5 * hx
    elif direction == "y":
        corner[1] += 0.5 * hy
    elif direction == "z":
        corner[2] += 0.5 * hz
    else:
        pass
    # grid coordinates / indices
    x0 = np.floor((P[:, 0] - corner[0]) / hx).astype(int)  # (N, )
    y0 = np.floor((P[:, 1] - corner[1]) / hy).astype(int)  # (N, )
    z0 = np.floor((P[:, 2] - corner[2]) / hz).astype(int)  # (N, )
    x1 = x0 + 1  # (N, )
    y1 = y0 + 1  # (N, )
    z1 = z0 + 1  # (N, )

    xd = (P[:, 0] - corner[0]) / hx - x0  # (N, )
    yd = (P[:, 1] - corner[1]) / hy - y0  # (N, )
    zd = (P[:, 2] - corner[2]) / hz - z0  # (N, )

    # data terms for the trilinear interpolation weight matrix
    weight_000 = (1 - xd) * (1 - yd) * (1 - zd)
    weight_100 = xd * (1 - yd) * (1 - zd)
    weight_010 = (1 - xd) * yd * (1 - zd)
    weight_110 = xd * yd * (1 - zd)
    weight_001 = (1 - xd) * (1 - yd) * zd
    weight_101 = xd * (1 - yd) * zd
    weight_011 = (1 - xd) * yd * zd
    weight_111 = xd * yd * zd
    data_term = np.concatenate((weight_000,
                                weight_100,
                                weight_010,
                                weight_110,
                                weight_001,
                                weight_101,
                                weight_011,
                                weight_111))

    row_idx = np.arange(P.shape[0])
    row_idx = np.tile(row_idx, 8)

    if direction == "x":
        num_grids = (nx - 1) * ny * nz
        staggered_grid_idx = np.arange((nx - 1) * ny * nz).reshape((nx - 1, ny, nz))
    elif direction == "y":
        num_grids = nx * (ny - 1) * nz
        staggered_grid_idx = np.arange(nx * (ny - 1) * nz).reshape((nx, ny - 1, nz))
    elif direction == "z":
        num_grids = nx * ny * (nz - 1)
        staggered_grid_idx = np.arange(nx * ny * (nz - 1)).reshape((nx, ny, nz - 1))
    else:
        num_grids = nx * ny * nz
        staggered_grid_idx = np.arange(nx * ny * nz).reshape((nx, ny, nz))

    col_idx_000 = staggered_grid_idx[x0, y0, z0]
    col_idx_100 = staggered_grid_idx[x1, y0, z0]
    col_idx_010 = staggered_grid_idx[x0, y1, z0]
    col_idx_110 = staggered_grid_idx[x1, y1, z0]
    col_idx_001 = staggered_grid_idx[x0, y0, z1]
    col_idx_101 = staggered_grid_idx[x1, y0, z1]
    col_idx_011 = staggered_grid_idx[x0, y1, z1]
    col_idx_111 = staggered_grid_idx[x1, y1, z1]
    col_idx = np.concatenate((col_idx_000,
                              col_idx_100,
                              col_idx_010,
                              col_idx_110,
                              col_idx_001,
                              col_idx_101,
                              col_idx_011,
                              col_idx_111))

    W = coo_matrix((data_term, (row_idx, col_idx)), shape=(P.shape[0], num_grids))
    return W


def poisson_surface_reconstruction(P, N, nx, ny, nz, padding, save_path):
    # bounding box size of the point cloud
    bbox_size = np.max(P, 0) - np.min(P, 0)

    # grid spacing along x, y, and z axis ZJA: divide the box into 64x64x64 small boxes
    hx = bbox_size[0] / nx
    hy = bbox_size[1] / ny
    hz = bbox_size[2] / nz

    # the world coordinates of the bottom left front corner of the volume, ZJA: pad a 'padding' number of cubes
    # at the bottom_left_front_corner
    bottom_left_front_corner = np.min(P, 0) - padding * np.array([hx, hy, hz])

    # grid numbers along x, y, and z axis, ZJA: I guess the program will pad in both side of one axis
    nx += 2 * padding
    ny += 2 * padding
    nz += 2 * padding

    # construct the system matrix
    G = fd_grad(nx, ny, nz, hx, hy, hz)  # matrix of gradient operator

    # construct the trilinear interpolation matrices along x, y, and z axis, in chinese: cha zhi, insert values
    Wx = trilinear_interpolation_weights(nx, ny, nz, bottom_left_front_corner, P, hx, hy, hz, direction="x")
    Wy = trilinear_interpolation_weights(nx, ny, nz, bottom_left_front_corner, P, hx, hy, hz, direction="y")
    Wz = trilinear_interpolation_weights(nx, ny, nz, bottom_left_front_corner, P, hx, hy, hz, direction="z")
    W = trilinear_interpolation_weights(nx, ny, nz, bottom_left_front_corner, P, hx, hy, hz)

    # distribute the normal vectors to staggered grids
    vx = Wx.T @ N[:, 0]
    vy = Wy.T @ N[:, 1]
    vz = Wz.T @ N[:, 2]
    v = np.concatenate([vx, vy, vz])
    print("Start solving for the characteristic function!")
    tic = time.time()
    g, _ = cg(G.T @ G, G.T @ v, maxiter=2000, tol=1e-5)  # G is the gradient operator, G.T@ G is laplacian operator
    toc = time.time()
    print(f"Linear solver finished! {toc - tic:.2f} sec")

    # align the zero iso surface with the input point cloud
    sigma = np.mean(W @ g)
    g -= sigma

    # extract the mesh using marching cubes
    g_field = g.reshape(nx, ny, nz)
    vertices, triangles = mcubes.marching_cubes(g_field, 0)

    # align the vertex coordinates with the input point cloud
    vertices[:, 0] = vertices[:, 0] * hx + bottom_left_front_corner[0]
    vertices[:, 1] = vertices[:, 1] * hy + bottom_left_front_corner[1]
    vertices[:, 2] = vertices[:, 2] * hz + bottom_left_front_corner[2]

    # save the mesh
    mcubes.export_obj(vertices, triangles, save_path)
    print(f"{save_path} saved")


if __name__ == '__main__':
    import open3d as o3d
    import argparse
    import os


    def file_path(string):
        if os.path.isfile(string):
            return string
        else:
            raise FileNotFoundError(string)


    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=file_path, default='data/wheel.ply')  # path
    parser.add_argument('--nx', type=int, default=64)  # nx,ny,nz, represents how many samples are in a set
    parser.add_argument('--ny', type=int, default=64)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--padding', type=int, default=8)
    par = parser.parse_args()

    data_dir = os.path.dirname(par.path)
    file_name = os.path.basename(par.path).split(".")[0]
    save_path = os.path.join(data_dir, f"PSR_nx_{par.nx}_ny_{par.ny}_nz_{par.nz}_" + file_name + ".obj")

    pcd = o3d.io.read_point_cloud(par.path)
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)

    poisson_surface_reconstruction(P, N, par.nx, par.ny, par.nz, par.padding, save_path)
