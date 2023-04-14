
#import packages yang diperlukan
from sklern.cluster import Kmeans
import matpoltlib.pyplot as plt
import argparse
import utils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i","--iamge",required= True,help="path to the image")
ap.add_argument("-c","--clusters",required= True,type=int,
                help="# of clusters")
args = vars(ap.parse_args())


image=cv2.imread(args["image"])
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

# Menunjukkan Gambar
plt.figure()
plt.axis("off")
plt.imshow(image)




#klasterisasi pixel
clt=KMeans(n_clusters=args["clusters"])
clt.fit(image)


hist=utils.centroid_histogram(clt)
bar=utils.plot_color(hist, clt.cluster_centers_)

#show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()