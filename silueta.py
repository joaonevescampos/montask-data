import os
import cv2
import numpy as np

input_root = "monstrix"           # pasta com suas imagens
output_root = "silueta-monstrix" # pasta de saída

for root, dirs, files in os.walk(input_root):

    # recria estrutura de pastas na saída
    relative_path = os.path.relpath(root, input_root)
    output_dir = os.path.join(output_root, relative_path)

    os.makedirs(output_dir, exist_ok=True)

    for file in files:

        if not file.lower().endswith(".png"):
            continue

        input_path = os.path.join(root, file)
        output_path = os.path.join(output_dir, file)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        # garantir que tem canal alpha
        if img.shape[2] == 3:
            b,g,r = cv2.split(img)
            alpha = np.full(b.shape, 255, dtype=np.uint8)
        else:
            b,g,r,alpha = cv2.split(img)

        # criar imagem branca
        white = np.full(b.shape, 255, dtype=np.uint8)

        # manter apenas onde existe pixel visível
        mask = alpha > 0

        new_b = np.zeros_like(b)
        new_g = np.zeros_like(g)
        new_r = np.zeros_like(r)

        new_b[mask] = 255
        new_g[mask] = 255
        new_r[mask] = 255

        result = cv2.merge((new_b, new_g, new_r, alpha))

        cv2.imwrite(output_path, result)

print("Silhuetas geradas com sucesso.")

