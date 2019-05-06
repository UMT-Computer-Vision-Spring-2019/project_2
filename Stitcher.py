from harris_response import *
import skimage.transform as skt


class Stitcher(object):

    def __init__(self, image_1, image_2):

        self.images = [image_1, image_2]

    def find_keypoints(self, image, n_keypoints):

        """
        Step 1: This method locates features that are "good" for matching.  To do this we will implement the Harris
        corner detector
        """

        filter_shape = (5, 5)

        # Compute smoothed harris response
        out = convolve(compute_harris_response(image, filter_shape),
                       Filter.make_gauss(filter_shape, 2))  # Smooth results

        # Find some good features to match
        # form: [u, v]
        x, y = anms_fast(h=out, n=n_keypoints)

        # Return the locations
        # form: [u, v]
        return x, y

    def generate_descriptors(self, img, l=21):
        """
        Step 2: After identifying relevant keypoints, we need to come up with a quantitative description of the
        neighborhood of that keypoint, so that we can match it to keypoints in other images.
        """
        u, v = self.find_keypoints(img, 100)

        ofs = l // 2

        d_out = []
        u_out = []
        v_out = []

        m = len(img)
        n = len(img[0])

        # check for u and v to be same dimensions
        for i in range(len(u)):

            c_x = v[i]
            c_y = u[i]

            # If we cannot get a description for key point, throw it out
            if c_x + ofs > m or c_x - ofs < 0 or c_y + ofs > n or c_y - ofs < 0:
                continue

            sub = img[v[i] - ofs: v[i] + ofs + 1, u[i] - ofs: u[i] + ofs + 1]
            if sub.shape[0] == l and sub.shape[1] == l:
                u_out.append(u[i])
                v_out.append(v[i])
                d_out.append(sub)

        return np.stack(d_out), np.asarray(u_out, dtype=int), np.asarray(v_out, dtype=int)

    def D_hat(self, d):
        return (d - d.mean()) / np.std(d)

    def error(self, d1, d2):
        return np.sum((d1 - d2) ** 2)

    def match_keypoints(self, r=0.7):
        """
        Step 3: Compare keypoint descriptions between images, identify potential matches, and filter likely
        mismatches
        """

        d1, u1, v1 = self.generate_descriptors(self.images[0], 0)
        d2, u2, v2 = self.generate_descriptors(self.images[1], 1)

        match_out = []
        value_list = []
        for i, D1 in enumerate(d1):

            smallest = np.inf
            index2_smallest = 0
            smallest2 = np.inf
            index2_smallest2 = 0
            D1_hat = (D1 - np.mean(D1)) / np.std(D1)
            value = 0
            value2 = 0

            for j, D2 in enumerate(d2):
                D2_hat = (D2 - np.mean(D2)) / np.std(D2)
                E = np.sum(np.square(D1_hat - D2_hat))
                if E < smallest:  # best match
                    smallest = E
                    value = E
                    index2_smallest = j
            np.delete(d1, index2_smallest, 0)
            np.delete(d2, index2_smallest, 0)

            for j, D2 in enumerate(d2):

                D2_hat = (D2 - np.mean(D2)) / np.std(D2)
                E = np.sum(np.square(D1_hat - D2_hat))

                if E < smallest2:  # the second best match
                    smallest2 = E
                    value2 = E
                    index2_smallest = j

            if value < (r * value2):
                match_out.append((u1[i], v1[i], u2[index2_smallest], v2[index2_smallest]))
                value_list.append(value)

        return np.asarray(match_out)

    def find_homography(self, uv, uv2):
        """
        Step 4: Find a linear transformation (of various complexities) that maps pixels from the second image to
        pixels in the first image
        """

        if uv.shape != uv2.shape:
            raise ValueError("X and X_prime must have matching shapes")
        if uv.shape[0] < 4:
            raise ValueError("Not enough points")

        # matches = np.column_stack(uv, uv2)

        A = np.zeros((2 * len(uv), 9))

        for i in range(len(uv)):
            A[2 * i, :] = [0, 0, 0, -uv[i, 0], -uv[i, 1], -1, uv2[i, 1] * uv[i, 0], uv2[i, 1] * uv[i, 1], uv2[i, 1]]
            A[2 * i + 1, :] = [uv[i, 0], uv[i, 1], 1, 0, 0, 0, -uv2[i, 0] * uv[i, 0], -uv2[i, 0] * uv[i, 1], -uv2[i, 0]]

        # print(A)
        U, Sigma, Vt = np.linalg.svd(A)

        H = Vt[-1, :].reshape((3, 3))
        H /= H[2, 2]

        return H

    def RANSAC(self, number_of_iterations=10, n=10, r=3, d=8):

        H_best = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        num_inliers = 0

        matches = self.match_keypoints()  # matches should be of the form [u1, v1, u2, v2]

        for i in range(number_of_iterations):
            # 1. Select a random sample of length n from the matches
            np.random.shuffle(matches)
            sub = matches[0:n, :]
            test = matches[n:, :]

            # 2. Compute a homography based on these points using the methods given above
            H = self.find_homography(sub[:, 0:2], sub[:, 2:])

            # 3. Apply this homography to the remaining points that were not randomly selected
            test_p = test[:, 0:2]
            test_p = np.column_stack((test_p, np.ones(len(test_p))))
            uv_p = (H @ test_p.T).T
            test_u = uv_p[:, 0] / uv_p[:, 2]
            test_v = uv_p[:, 1] / uv_p[:, 2]

            # 4. Compute the residual between observed and predicted feature locations
            R = np.zeros_like(test_u)
            for i in range(len(test_p)):
                R[i] = np.sqrt((test_u[i] - test[i, 2]) ** 2 + (test_v[i] - test[i, 3]) ** 2)

            # 5. Flag predictions that lie within a predefined distance r from observations as inliers
            inl = np.zeros_like(R)
            for i in range(len(inl)):
                if R[i] < r:
                    inl[i] = 1
                else:
                    inl[i] = 0
            num_inl = np.sum(inl)

            # 6. If number of inliers is greater than the previous best
            #    and greater than a minimum number of inliers d,
            #    7. update H_best
            #    8. update list_of_inliers
            if num_inl > num_inliers:
                if num_inl > d:
                    H_best = H
                    num_inliers = num_inl

        return H_best, num_inliers

    def stitch(self):
        """
        Step 5: Transform second image into local coordinate system of first image, and (perhaps) perform blending
        to avoid obvious seams between images.
        """
        H_best = self.RANSAC(10, 10, 3, 8)

        im1 = self.images[0]
        im2 = self.images[1]

        transform = skt.ProjectiveTransform(H_best)
        im_2_warped = skt.warp(im2, transform, output_shape=(im1.shape[0], im1.shape[1] + (int(im1.shape[1] * 0.4))))

        im1t = np.zeros_like(im_2_warped)

        for v in range(im1.shape[0]):
            for u in range(im1.shape[1]):
                if im1[v, u] != 0:
                    im1t[v, u] = im1[v, u]

        img_out = np.zeros_like(im_2_warped)

        for v in range(img_out.shape[0]):
            for u in range(img_out.shape[1]):
                if im1t[v, u] == 0 and im_2_warped[v, u] == 0:
                    img_out[v, u] = 0

                elif im1t[v, u] != 0 and im_2_warped[v, u] == 0:
                    img_out[v, u] = im1[v, u]
                elif im1t[v, u] == 0. and im_2_warped[v, u] != 0:
                    img_out[v, u] = im_2_warped[v, u]
                else:
                    img_out[v, u] = (im_2_warped[v, u] + im1[v, u]) / 2

        return img_out
