import cv2
import numpy as np

def sift_matching(image1, image2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    kp_image1 = cv2.drawKeypoints(image1, keypoints1, None)
    kp_image2 = cv2.drawKeypoints(image2, keypoints2, None)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Matches Before Ransac
    out0 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)
    num_matches_raw = len(matches)
    print(f"Total Number of Raw Matches:{num_matches_raw}")

    # Apply RANSAC to filter out outliers
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel()

    # Filter matches based on RANSAC mask
    good_matches = [m for i, m in enumerate(matches) if mask[i]]
    num_matches_ransac = len(good_matches)
    out = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    print(f"Total Number of RANSAC Matches :{num_matches_ransac} ")

    # Display warp images
    img1 = image1
    M,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    warped_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

    return (
        kp_image1,
        kp_image2,
        out0,
        out,
        {
            "number raw matches": num_matches_raw,
            "number ransac matches": num_matches_ransac,
        },
        img1,
        warped_image
       
    )


def akaze_matching(image1, image2):
    akaze = cv2.AKAZE_create()
    keypoints1, descriptors1 = akaze.detectAndCompute(image1, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp_image1 = cv2.drawKeypoints(image1, keypoints1, None)
    kp_image2 = cv2.drawKeypoints(image2, keypoints2, None)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Matches Before Ransac
    out0 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)
    num_matches_raw = len(matches)
    print(f"Total Number of Raw Matches:{num_matches_raw}")

    # Apply RANSAC to filter out outliers
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel()

    # Filter matches based on RANSAC mask
    good_matches = [m for i, m in enumerate(matches) if mask[i]]
    num_matches_ransac = len(good_matches)
    out = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    print(f"Total Number of RANSAC Matches :{num_matches_ransac} ")

    # Display warp images
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    img1 = image1
    warped_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

    return (kp_image1,
        kp_image2,
        out0,
        out,
        {
            "number raw matches": num_matches_raw,
            "number ransac matches": num_matches_ransac,
        },
        img1,
        warped_image
        )



def orb_matching(image1, image2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    kp_image1 = cv2.drawKeypoints(image1, keypoints1, None)
    kp_image2 = cv2.drawKeypoints(image2, keypoints2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Matches Before Ransac
    out0 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)
    num_matches_raw = len(matches)
    print(f"Total Number of Raw Matches:{num_matches_raw}")
    # Apply RANSAC to filter out outliers
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel()

    # Filter matches based on RANSAC mask
    good_matches = [m for i, m in enumerate(matches) if mask[i]]
    num_matches_ransac = len(good_matches)
    out = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    print(f"Total Number of RANSAC Matches :{num_matches_ransac} ")
    
    # Display warp images
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    img1 = image1
    warped_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

    return (kp_image1,
        kp_image2,
        out0,
        out,
        {
            "number raw matches": num_matches_raw,
            "number ransac matches": num_matches_ransac,
        },
        img1,
        warped_image,

        )

    

def surf_matching(image1, image2):
    surf = cv2.SURF_create()
    keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(image2, None)
    kp_image1 = cv2.drawKeypoints(image1, keypoints1, None)
    kp_image2 = cv2.drawKeypoints(image2, keypoints2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Matches Before Ransac
    out0 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)
    num_matches_raw = len(matches)
    print(f"Total Number of Raw Matches:{num_matches_raw}")
    # Apply RANSAC to filter out outliers
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel()

    # Filter matches based on RANSAC mask
    good_matches = [m for i, m in enumerate(matches) if mask[i]]
    num_matches_ransac = len(good_matches)
    print(f"Total Number of RANSAC Matches :{num_matches_ransac} ")

    out = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    img1 = image1
    warped_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))
    return (
        kp_image1,
        kp_image2,
        out0,
        out,
        {
            "number raw matches": num_matches_raw,
            "number ransac matches": num_matches_ransac,
        },
        img1,
        warped_image
    )
def brisk_matching(image1, image2):
    brisk = cv2.BRISK_create()
    keypoints1, descriptors1 = brisk.detectAndCompute(image1, None)
    keypoints2, descriptors2 = brisk.detectAndCompute(image2, None)
    kp_image1 = cv2.drawKeypoints(image1, keypoints1, None)
    kp_image2 = cv2.drawKeypoints(image2, keypoints2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Matches Before Ransac
    out0 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)
    num_matches_raw = len(matches)
    print(f"Total Number of Raw Matches:{num_matches_raw}")
    # Apply RANSAC to filter out outliers
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel()

    # Filter matches based on RANSAC mask
    good_matches = [m for i, m in enumerate(matches) if mask[i]]
    num_matches_ransac = len(good_matches)
    print(f"Total Number of RANSAC Matches :{num_matches_ransac} ")
    
    out = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    img1 = image1
    warped_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))
    return (
        kp_image1,
        kp_image2,
        out0,
        out,
        {
            "number raw matches": num_matches_raw,
            "number ransac matches": num_matches_ransac,
        },
        img1,
        warped_image
    )

def kaze_matching(image1, image2):
    kaze = cv2.KAZE_create()
    keypoints1, descriptors1 = kaze.detectAndCompute(image1, None)
    keypoints2, descriptors2 = kaze.detectAndCompute(image2, None)
    kp_image1 = cv2.drawKeypoints(image1, keypoints1, None)
    kp_image2 = cv2.drawKeypoints(image2, keypoints2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Matches Before Ransac
    out0 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)
    num_matches_raw = len(matches)
    print(f"Total Number of Raw Matches:{num_matches_raw}")
    # Apply RANSAC to filter out outliers
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel()

    # Filter matches based on RANSAC mask
    good_matches = [m for i, m in enumerate(matches) if mask[i]]
    num_matches_ransac = len(good_matches)
    print(f"Total Number of RANSAC Matches :{num_matches_ransac} ")
    
    out = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    img1 = image1
    warped_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))
    return (
        kp_image1,
        kp_image2,
        out0,
        out,
        {
            "number raw matches": num_matches_raw,
            "number ransac matches": num_matches_ransac,
        },
        img1,
        warped_image
    )


def matching_function(image1, image2, method):
    image1 = cv2.resize(image1, (500, 500))
    image2 = cv2.resize(image2, (500, 500))

    if method == 'SIFT':
        matches = sift_matching(image1, image2)
    elif method == 'ORB':
        matches = orb_matching(image1, image2)
    elif method == 'AKAZE':
        matches = akaze_matching(image1, image2)
    elif method == 'BRISK':
        matches = brisk_matching(image1, image2)
    elif method == 'SURF':
        matches = surf_matching(image1, image2)
    elif method == 'KAZE':
        matches = kaze_matching(image1, image2)

    return matches











   