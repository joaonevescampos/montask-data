import cv2
import os
import numpy as np
import shutil

input_folder = "entrada"
output_folder = "saida"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

count = 0
scale = 2

for file in os.listdir(input_folder):

    if not (file.endswith(".png") or file.endswith(".jpg")):
        continue

    path = os.path.join(input_folder, file)
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    data = []

    for c in contours:

        area = cv2.contourArea(c)

        if area < 150:
            continue

        x, y, w, h = cv2.boundingRect(c)

        data.append((c, x, y, w, h))

    # ordenar por Y e depois por X
    data.sort(key=lambda d: (d[2], d[1]))

    # estimar altura média
    heights = [d[4] for d in data]
    avg_height = np.mean(heights) if heights else 0

    rows = []
    current_row = []
    last_y = None

    for item in data:

        c, x, y, w, h = item

        if last_y is None:
            current_row.append(item)
            last_y = y
            continue

        if abs(y - last_y) < avg_height * 0.6:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]

        last_y = y

    if current_row:
        rows.append(current_row)

    # ordenar cada linha pela posição X
    for row in rows:
        row.sort(key=lambda d: d[1])

    # pasta da imagem
    image_folder = os.path.join(output_folder, os.path.splitext(file)[0])
    os.makedirs(image_folder)

    for row_index, row in enumerate(rows):

        row_folder = os.path.join(image_folder, f"linha_{row_index+1}")
        os.makedirs(row_folder)

        for item in row:

            c, x, y, w, h = item

            sprite = img[y:y+h, x:x+w]

            mask = np.zeros((h, w), dtype=np.uint8)

            shifted = c - [x, y]

            cv2.drawContours(mask, [shifted], -1, 255, -1)

            kernel = np.ones((3,3), np.uint8)

            mask = cv2.erode(mask, kernel, iterations=1)

            mask = cv2.GaussianBlur(mask, (5,5), 0)

            sprite = cv2.cvtColor(sprite, cv2.COLOR_BGR2BGRA)

            sprite[:,:,3] = mask

            b,g,r,a = cv2.split(sprite)

            transparent = a == 0

            kernel = np.ones((3,3), np.uint8)

            b_d = cv2.dilate(b, kernel)
            g_d = cv2.dilate(g, kernel)
            r_d = cv2.dilate(r, kernel)

            b[transparent] = b_d[transparent]
            g[transparent] = g_d[transparent]
            r[transparent] = r_d[transparent]

            sprite = cv2.merge((b,g,r,a))

            sprite = cv2.resize(
                sprite,
                (w * scale, h * scale),
                interpolation=cv2.INTER_LANCZOS4
            )

            # filtro de nitidez (sharpen)
            # kernel = np.array([
            #     [0,-1,0],
            #     [-1,5,-1],
            #     [0,-1,0]
            # ])

            # sprite = cv2.filter2D(sprite, -1, kernel)

            filename = f"monster_{count}.png"

            cv2.imwrite(
                os.path.join(row_folder, filename),
                sprite
            )

            count += 1

print("Sprites extraídos:", count)