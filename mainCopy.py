import cv2 as cv
import imutils as imutils
import numpy as np
import matplotlib.pyplot as plt

SHOW = False
MIN_MATCH_COUNT = 3

def rotate(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray_image = cv.GaussianBlur(gray_image, (3, 3), 0)

    ret, th = cv.threshold(gray_image, 127, 255, cv.THRESH_TRUNC)

    edged_image = cv.Canny(th, 4, 10, 5)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    closed = cv.morphologyEx(edged_image, cv.MORPH_CLOSE, kernel)
    edged_image = closed

    all_contours = cv.findContours(edged_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    all_contours = imutils.grab_contours(all_contours)

    all_contours = sorted(all_contours, key=cv.contourArea, reverse=True)[:1]

    if not all_contours:
        return image, None, None

    max_contour_len = cv.contourArea(all_contours[0])
    max_contour = all_contours[0]

    if max_contour is None:
        return image, None, None

    perimeter = cv.arcLength(max_contour, True)
    ROIdimensions = cv.approxPolyDP(max_contour, 0.02 * perimeter, True)

    # cv.drawContours(image, [ROIdimensions], -1, (0,255,0), 2)

    if ROIdimensions.shape[0] == 4:
        ROIdimensions = ROIdimensions.reshape(4, 2)
    else:
        return image, None, None

    rect = np.zeros((4, 2), dtype=np.float32)

    s = np.sum(ROIdimensions, axis=1)
    rect[0] = ROIdimensions[np.argmin(s)]
    rect[2] = ROIdimensions[np.argmax(s)]

    diff = np.diff(ROIdimensions, axis=1)
    rect[1] = ROIdimensions[np.argmin(diff)]
    rect[3] = ROIdimensions[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
    widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype=np.float32)

    transformMatrix = cv.getPerspectiveTransform(rect, dst)

    inversed = cv.getPerspectiveTransform(dst, rect)
    scan = cv.warpPerspective(image, transformMatrix, (maxWidth, maxHeight))

    return image, scan, inversed


def get_wide_image(image, background=[0, 0, 255], delta=10):
    n, m, *_ = image.shape
    wide_image = np.full((n + 2 * delta, m + 2 * delta, 3), background, dtype=image.dtype)
    # wide_image.fill(np.array(background, dtype=image.dtype))

    for y in range(n):
        for x in range(m):
            wide_image[y + delta, x + delta] = image[y, x]
    return wide_image

def arrToInt(arr):
    return [int(i) for i in arr]



def get_matched_coordinates(temp_img, map_img):
    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(temp_img, None)
    kp2, des2 = sift.detectAndCompute(map_img, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = temp_img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        map_img = cv.polylines(
            map_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        raise Exception('Not enough matches are found')

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv.drawMatches(temp_img, kp1, map_img, kp2,
                           good, None, **draw_params)

    if SHOW:
        plt.imshow(img3, 'gray'), plt.show()

    # result image path
    # cv.imwrite('./result.png', img3)

    return dst

COEF = 0.15

def sub(arr1, arr2):
    res = []
    for i in range(len(arr1)):
        res.append(arr1[i] - arr2[i])
    return res

def getBorderCoords(coords,final):
    maxX, maxY, minX, minY = 0, 0, final.shape[0], final.shape[1]
    for c in coords:
        maxX = max(maxX, c[0][0])
        maxY = max(maxY, c[0][1])
        minX = min(minX, c[0][0])
        minY = min(minY, c[0][1])
    return (maxX, maxY, minX, minY)



def process_image(full_image_fn,puzzle_fn):
    #full_image_fn = 'DOG1.jpg'
    #puzzle_fn = 'DOG1_PUZZLE.jpg'

    full_image = cv.imread(full_image_fn)
    final = full_image.copy()
    gray_final = cv.cvtColor(full_image, cv.COLOR_BGR2GRAY)

    img = cv.imread(puzzle_fn)
    original = img.copy()
    canny = cv.Canny(img, 120, 255, 1)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv.dilate(canny, kernel, iterations=1)
    cnts = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    image_number = 0
    pieces = []
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = str(image_number + 1)
        frameSize = min(w, h)
        shift = frameSize / 6
        textPoint = [x + w / 2 - shift * len(text),
                     y + h / 2 + shift * len(text)]

        cv.putText(img, text, arrToInt(textPoint), cv.FONT_HERSHEY_PLAIN, frameSize / 24, (255, 0, 0), 3)

        pieces.append(original[y:y + h, x:x + w])
        # cv.imshow('piece')
        wide_image = get_wide_image(pieces[image_number])
        pieces[image_number], scan, *_ = rotate(wide_image.copy())
        if scan is not None:
            pieces[image_number] = scan
        image_number += 1

    puzzle_res_name = puzzle_fn.split('.')
    puzzle_res_name.insert(-1, '_pres.')
    cv.imwrite(''.join(puzzle_res_name), img)

    found_places = []
    sum_len = 0
    for i in range(len(pieces)):
        template = pieces[i][int(COEF * pieces[0].shape[0]):int((1 - COEF) * pieces[0].shape[0]),
                   int(COEF * pieces[0].shape[1]):int((1 - COEF) * pieces[0].shape[1])]

        gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        try:
            coords = get_matched_coordinates(gray_template, gray_final)
        except Exception:
            continue

        maxX, maxY, minX, minY = getBorderCoords(coords,final)
        l = (maxX - minX + maxY - minY) / 2
        if l < 0.5 * max(final.shape) and abs(minX) < abs(maxX) and abs(minY) < abs(maxY) and maxX < final.shape[
            0] + 20 and maxY < final.shape[1] + 20:
            found_places.append((i, maxX, maxY, minX, minY))
            sum_len += l

    avg_len = sum_len / len(found_places)

    for i in range(len(found_places)):
        index, maxX, maxY, minX, minY = found_places[i]
        l = (maxX - minX + maxY - minY) / 2
        if l > 0.666 * avg_len and l < 1.5 * avg_len:
            cv.rectangle(final, (int(minX), int(minY)), (int(maxX), int(maxY)), (255, 0, 0), 3)

            frameSize = min(maxX - minX, maxY - minY)

            text = str(index + 1)

            # textPoint1 = [(coords[0][0][i] + coords[2][0][i])/2 for i in range(2)]
            # textPoint2 = [(coords[1][0][i] + coords[3][0][i])/2 for i in range(2)]
            # shift = frameSize/6
            textPoint = [(maxX + minX) / 2 - shift * len(text), (maxY + minY) / 2 + shift * len(text)]

            cv.putText(final, text, arrToInt(textPoint), cv.FONT_HERSHEY_PLAIN, frameSize / 24, (255, 0, 0), 3)

    res_name = puzzle_fn.split('.')
    res_name.insert(-1, '_res.')
    last_image_fn = ''.join(res_name)
    cv.imwrite(last_image_fn, final)
    last_puzzle_name =''.join(puzzle_res_name)
    return [last_image_fn,last_puzzle_name]


