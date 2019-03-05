from convolution import *
from Filter import *
import numba


def compute_harris_response(img, filter_shape):

    gauss = Filter.make_gauss(shape=filter_shape, sigma=2)
    img = convolve(img, gauss)

    # NOTE: (*) = convolution

    # Make respective Sobel filters for computing gradients
    S_u = Filter.make_sobel_u()
    S_v = Filter.make_sobel_v()

    # Compute gradient of image with respect to u = I_u
    di_du = convolve(img, S_u)

    # Compute gradient of image with respect to v = I_v
    di_dv = convolve(img, S_v)

    # Compute squares of gradient
    # I_uu = w (*) np.multiply(I_u, I_u)
    di_du_sq = convolve(di_du ** 2, gauss)
    di_dv_sq = convolve(di_dv ** 2, gauss)

    # Compute product of gradients
    di_du_di_dv = convolve(np.multiply(di_du, di_dv), gauss)

    # Finally, compute the harris response
    # Note: we add the 1e-16 to prevent division by 0.
    # 1.0 + 1e-16 = 1.0, so we do not have to worry about screwing up division
    return np.divide((np.multiply(di_du_sq, di_dv_sq) - di_du_di_dv ** 2), (di_du_sq + di_dv_sq) + 1e-16)


def non_maximal_suppression(h, n=100, max_filter_size=3):
    """Returns two vectors, one for x point location, one for y point location.
    These points correspond to local maxima point locations"""

    local_max_pts = local_max_loc_and_intensity(h)

    # Filter out local maxima with relatively weak responses
    local_max_pts = local_max_pts[local_max_pts[:, 2] > 1e-5]

    # Sort by intensity and reverse
    local_max_pts = np.flip(local_max_pts[local_max_pts[:, 2].argsort(kind='mergesort')], axis=0)

    # Keep best n
    return local_max_pts[:n, 1], local_max_pts[:n, 0]


def local_max_loc_and_intensity(h):

    """
    Helper function to compute locations of local maxima
    :param h: harris repsonse
    :param max_filter_size: size of filter window to be used
    :return: matrix that is shape of number of local maxima.
    First column is i point location, second column is j point location
    Last column is value of harris response at that location
    """
    bool_mask = non_linear_convolve(h, 3)

    # Get list of points where mask is true
    i, j = np.where(bool_mask == True)

    # Grab the corresponding intensities
    intensities = h[i, j]

    # (n,) -> (n, 1)
    i = np.expand_dims(i, axis=1)
    j = np.expand_dims(j, axis=1)
    intensities = np.expand_dims(intensities, axis=1)

    # form: [v, u, intensities]
    return np.hstack((i, j, intensities))


@numba.jit(nopython=True)
def point_distance_sq(p1, p2):
    """Computes squared euclidean distance between p1 and p2"""

    diff_x = (float(p1[0]) - float(p2[0]))
    diff_y = (float(p1[1]) - float(p2[1]))

    return (diff_x * diff_x) + (diff_y * diff_y)


def adaptive_non_maximal_suppression(h, n, fname=None, c=0.9):

    # pt = point
    # curr = current
    # loc = location
    # idx = index
    # dist = distance
    # num = number
    # pot = potential
    # kpts = key points

    print('running ANMS on', fname)

    max_loc_and_intensity = pre_process_loc_and_intensity(local_max_loc_and_intensity(h))

    local_max_pts_loc = max_loc_and_intensity[:, 0:2]
    local_max_pts_intensity = max_loc_and_intensity[:, 2]

    num_pot_kpts = len(local_max_pts_loc)

    # this list has index in the closest point w/ higher harris response
    closest_pt_loc = np.empty(shape=(num_pot_kpts, 2), dtype=np.int)
    closest_pt_dist = np.empty(shape=(num_pot_kpts, 1))
    closest_pt_dist.fill(np.inf)

    # for each point, find closest point w/ higher harris response
    for i in range(num_pot_kpts):
        for j in range(num_pot_kpts):

            # Don't want to compare point to itself
            if i == j:
                continue

            # Only compute distance if response is stronger(thanks to Dr. Douglas Brinkerhoff)
            elif local_max_pts_intensity[j] * c > local_max_pts_intensity[i]:

                new_dist = point_distance_sq(local_max_pts_loc[i], local_max_pts_loc[j])

                # If new point has higher intensity and is closer, record it
                if new_dist < closest_pt_dist[i]:
                    closest_pt_loc[i] = local_max_pts_loc[j]
                    closest_pt_dist[i] = new_dist

    # If dist is still infinity, it is the 'best' local max
    cond = closest_pt_dist[:, 0] == np.inf
    closest_pt_loc[cond] = local_max_pts_loc[cond]

    # Create matrix of [point_location_j, point_location_i, dist to closest point]
    pt_dist_mtx = np.hstack((closest_pt_loc, closest_pt_dist))

    pt_dist_mtx = np.flipud(pt_dist_mtx[pt_dist_mtx[:, 2].argsort(kind='mergesort')])

    points_to_keep = set()

    for i in range(len(pt_dist_mtx)):

        current_point = (int(pt_dist_mtx[i, 1]), int(pt_dist_mtx[i, 0]))

        if current_point not in points_to_keep:
            points_to_keep.add(current_point)

            if len(points_to_keep) == n:
                break

    print('ANMS finished', fname)

    # Save these points, b/c this process is very slow
    if fname:
        pts_to_keep_mtx = np.asarray(list(points_to_keep))
        np.save(fname, pts_to_keep_mtx)

    return zip(*points_to_keep)


@numba.jit(nopython=True, parallel=True)
def anms_helper(local_max_pts_loc, local_max_pts_intensity, c):

    num_pot_kpts = len(local_max_pts_loc)

    # this list has index in the closest point w/ higher harris response
    closest_pt_loc = np.zeros(shape=(num_pot_kpts, 2), dtype=np.int64)
    closest_pt_dist = np.zeros(shape=(num_pot_kpts, 1))
    closest_pt_dist.fill(np.inf)

    # for each point, find closest point w/ higher harris response
    for i in range(num_pot_kpts):
        for j in range(num_pot_kpts):

            # Don't want to compare point to itself
            if i == j:
                continue

            # Only compute distance if response is stronger(thanks to Dr. Douglas Brinkerhoff)
            elif local_max_pts_intensity[j] * c > local_max_pts_intensity[i]:

                new_dist = point_distance_sq(local_max_pts_loc[i], local_max_pts_loc[j])

                # If new point has higher intensity and is closer, record it
                if new_dist < closest_pt_dist[i, 0]:
                    closest_pt_loc[i] = local_max_pts_loc[i]
                    closest_pt_dist[i] = new_dist

    return closest_pt_loc, closest_pt_dist


def pre_process_loc_and_intensity(max_loc_and_intensity):

    # Filter out local maxima with relatively weak responses
    max_loc_and_intensity = max_loc_and_intensity[max_loc_and_intensity[:, 2] > 1e-3]

    # Sort by strength of harris response
    # Note: argsort sorts in ascending order(ie: -2, -1, 0, 1, 2, ..., inf)
    # but we want descending order, so we flip to reverse row order.
    max_loc_and_intensity = np.flipud(max_loc_and_intensity[max_loc_and_intensity[:, 2].argsort(kind='mergesort')])

    # Keep top 50%(this could be converted to a percentage,
    # just being lazy.(could also pick number / percentage of points to throw
    # out and then do so by randomly sampling from indices, this has
    # added benefit of [theoretically] keeping the same spatial
    # distribution as the original points(thanks to Dr. Douglas Brinkerhoff))
    num_to_keep = len(max_loc_and_intensity) // 2

    return max_loc_and_intensity[:num_to_keep, :]


def anms_fast(h, n, c=0.9):

    # pt = point
    # curr = current
    # loc = location
    # idx = index
    # dist = distance
    # num = number
    # pot = potential
    # kpts = key points

    # Compute and pre-process local maxima
    # form: [v, u, intensities]
    max_loc_and_intensity = pre_process_loc_and_intensity(local_max_loc_and_intensity(h))

    local_max_pts_loc = max_loc_and_intensity[:, 0:2]
    local_max_pts_intensity = max_loc_and_intensity[:, 2]

    closest_pt_loc, closest_pt_dist = anms_helper(local_max_pts_loc, local_max_pts_intensity, c)

    # If dist is still infinity, it is the 'best' local max
    cond = closest_pt_dist[:, 0] == np.inf
    closest_pt_loc[cond] = local_max_pts_loc[cond]

    # Create matrix of [point_location_j, point_location_i, dist to closest point]
    pt_dist_mtx = np.hstack((closest_pt_loc, closest_pt_dist))

    pt_dist_mtx = np.flipud(pt_dist_mtx[pt_dist_mtx[:, 2].argsort(kind='mergesort')])

    points_to_keep = set()

    for i in range(len(pt_dist_mtx)):

        current_point = (int(pt_dist_mtx[i, 1]), int(pt_dist_mtx[i, 0]))

        if current_point not in points_to_keep:
            points_to_keep.add(current_point)

            if len(points_to_keep) == n:
                break

    # form: [u, v, intensities]
    return zip(*points_to_keep)
