import cv2 as cv
import numpy as np
from collections import deque

def dist_to(p1, p2):
    return np.linalg.norm(p1.reshape(-1, 1, 2) - p2.reshape(1, -1, 2), axis=-1)

def in_bound(x, y, size):
    if x >= 0 and y >= 0 and x < size and y < size:
        return True
    return False

def merge_corner(corners, img_size, neighbor_size=2):
    map = np.zeros((img_size, img_size))
    map[corners[:, 0], corners[:, 1]] = 1
    visited = np.zeros(corners.shape[0])

    d_ys = list(range(-neighbor_size, neighbor_size + 1))
    d_xs = list(range(-neighbor_size, neighbor_size + 1))

    corners_tuple = [tuple(p) for p in corners]
    cnt = 1
    for i in range(corners.shape[0]):
        if visited[i]:
            continue
        q = deque([i])
        while len(q) > 0:
            cur_idx = q.pop()
            p = corners[cur_idx]
            y, x = p
            visited[cur_idx] = cnt
            for d_y in d_ys:
                for d_x in d_xs:
                    n_y = y + d_y
                    n_x = x + d_x
                    if not in_bound(n_x, n_y, img_size):
                        continue
                    n_p = (n_y, n_x)
                    if n_p in corners_tuple:
                        next_idx = corners_tuple.index(n_p)
                        if visited[next_idx]:
                            continue
                        q.append(next_idx)
        cnt += 1

    centers = []
    for i in range(1, cnt):
        points = corners[visited==i]
        center = points.mean(axis=0)
        centers.append(center)
    centers = np.vstack(centers)
    centers = np.round(centers).astype(np.int64)
    return centers

def is_edge(edge_img, edge_from, edge_to):
    diff = edge_to - edge_from
    n_steps = np.abs(diff).max()
    delta = diff / n_steps
    smoothed_edge_img = cv.blur(edge_img, ksize=(5,5))
    for i in range(n_steps):
        point = edge_from + (i + 1) * delta
        point = np.round(point).astype(np.int)
        y, x = point
        if not in_bound(x, y, edge_img.shape[0]):
            return False
        if smoothed_edge_img[y, x] == 0:
            return False
    return True

def trace_contour(edge_img, corners):
    edge_points = np.argwhere(edge_img)
    dist = dist_to(corners, edge_points)
    edge_nearest = dist.argmin(axis=-1)
    corners = edge_points[edge_nearest]

    visited = np.zeros(corners.shape[0])
    contour = []
    cur_idx = 0
    found = True
    while found:
        visited[cur_idx] = 1
        contour.append(cur_idx)
        found = False
        for j in range(corners.shape[0]):
            if visited[j]:
                continue
            if is_edge(edge_img, corners[cur_idx], corners[j]):
                cur_idx = j
                found = True
                break

    contour = np.array(contour)
    return contour

def get_edge(contour):
    return np.vstack([np.array([contour[i], contour[(i + 1) % contour.shape[0]]]) for i in range(contour.shape[0])])

def get_normal(gray, edge_img, edges, corners):
    gray = gray.astype(np.float)
    img_grad_y = cv.Sobel(gray, -1, 0, 1)
    img_grad_x = cv.Sobel(gray, -1, 1, 0)
    edge_points = np.argwhere(edge_img)
    normals = []
    for edge in edges:
        idx_from, idx_to = edge
        edge_from, edge_to = corners[idx_from], corners[idx_to]
        mid = (edge_from + edge_to) // 2

        dist = dist_to(mid.reshape(1, -1), edge_points)
        edge_nearest = dist.argmin(axis=-1)[0]
        mid = edge_points[edge_nearest]

        grad_x = img_grad_x[mid[0], mid[1]]
        grad_y = img_grad_y[mid[0], mid[1]]
        normal = np.array([grad_y, grad_x])
        normal /= (np.linalg.norm(normal) + 1e-6)
        normals.append(normal)

    normals = np.vstack(normals)
    return normals

def draw_corner(src, corners):
    img = src.copy()

    corners_img = np.zeros((src.shape[:2]))
    corners_img[corners[:, 0], corners[:, 1]] = 255
    corners_img = cv.dilate(corners_img, None)
    img[corners_img > 0] = [0, 0, 255]

    cv.imwrite('demo_corner.png', img)

def draw_contour(src, contour, corners):
    img = src.copy()

    for i in range(contour.shape[0]):
        p_from = corners[contour[i]]
        p_to = corners[contour[(i + 1) % contour.shape[0]]]
        cv.line(img, [p_from[1], p_from[0]], [p_to[1], p_to[0]], color=[0, 0, 255], thickness=3)

    cv.imwrite('demo_contour.png', img)

def draw_normal(src, edges, normals, line_scale=0.02):
    img = src.copy()
    line_length = line_scale * src.shape[0]
    for i in range(edges.shape[0]):
        edge = edges[i]
        normal = normals[i]

        idx_from, idx_to = edge
        edge_from, edge_to = corners[idx_from], corners[idx_to]
        mid = (edge_from + edge_to) // 2

        line_from = mid
        line_to = np.round(mid + normal * line_length).astype(np.int)
        cv.line(img, [line_from[1], line_from[0]], [line_to[1], line_to[0]], color=[0, 0, 255], thickness=3)

    cv.imwrite('demo_normal.png', img)

def process_floor_map(src, size=None):
    # for faster processing; may introduce inaccuraccy
    if size is not None:
        if isinstance(size, (int, float)):
            size = (size, size)
        src = cv.resize(src, size)

    if len(src.shape) == 3:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    corners = cv.cornerHarris(gray, 5, 3, 0.04)
    corners = np.argwhere(corners > 0.01 * corners.max())
    corners = merge_corner(corners, gray.shape[0])

    edge_img = cv.Canny(src, 50, 150)
    contour = trace_contour(edge_img, corners)
    edges = get_edge(contour)

    normals = get_normal(gray, edge_img, edges, corners)

    return corners, contour, edges, normals

def visualize(src, corners, contour, edges, normals):
    draw_corner(src, corners)
    draw_contour(src, contour, corners)
    draw_normal(src, edges, normals)

if __name__ == '__main__':
    src = cv.imread('floor_map.png')
    corners, contour, edges, normals = process_floor_map(src)
    visualize(src, corners, contour, edges, normals)