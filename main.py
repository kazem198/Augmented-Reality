import cv2
import numpy as np

cap = cv2.VideoCapture(0)

vid = cv2.VideoCapture("./AIFF.mp4")
target = cv2.imread('./target.jpg')          # queryImage
main = cv2.imread('./main.jpg')  # trainImage
car = cv2.imread('./Cars15.png')
h, w, _ = target.shape
frameNum = 1


car = cv2.resize(car, (w, h))
orb = cv2.ORB_create(nfeatures=1000)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(target, None)


# BFMatcher with default params
bf = cv2.BFMatcher()


# Apply ratio test


# cv2.imshow("mask", mask)
# cv2.imshow("fillnot", fillnot)
# cv2.imshow("finall", finall)
while True:
    # success, img = cap.read()
    img = main
    imgCopy = main.copy()
    kp2, des2 = orb.detectAndCompute(img, None)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    img3 = cv2.drawMatchesKnn(target, kp1, img, kp2, [good],
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv.drawMatchesKnn expects list of lists as matches.
    if len(good) > 10:
        if (frameNum == vid.get(cv2.CAP_PROP_FRAME_COUNT)):
            frameNum = 0
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # vid.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        _, vi = vid.read()

        vi = cv2.resize(vi, (w, h))

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]
                         ).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(
            img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        imgWrap = cv2.warpPerspective(
            vi, M, (main.shape[1], main.shape[0]))

        mask = np.zeros((main.shape[0], main.shape[1]), np.uint8)

        cv2.fillPoly(mask, [np.int32(dst)], (255, 255, 255))
        maskNot = cv2.bitwise_not(mask)

        andImg = cv2.bitwise_and(imgCopy, imgCopy, mask=maskNot)
        andImg = cv2.bitwise_or(imgWrap, andImg)
        cv2.imshow("andImg", andImg)
        cv2.imshow("maskNot", maskNot)
        cv2.imshow("imgwrap", imgWrap)
        frameNum += 1

    else:
        print("Not enough matches are found - {}/{}", format(len(good)))
        matchesMask = None
        # plt.imshow(img3),plt.show()
        cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
