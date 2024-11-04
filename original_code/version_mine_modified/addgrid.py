import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Carica l'immagine SVG
img = mpimg.imread('original_code/version_mine_modified/valLoss_trainLoss_resnet50isic2017complete.svg')
plt.imshow(img)

# Aggiungi la griglia
plt.grid(True, linestyle='--', color='gray', alpha=0.5)

# Mostra il grafico
plt.show()