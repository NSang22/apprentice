import cv2, numpy as np, urllib.request as u
r = u.urlopen("http://127.0.0.1:8771/frame")
h, w, c = map(int, r.headers["X-Shape"].split(","))
cv2.imwrite("logs/board.png", np.frombuffer(r.read(), np.uint8).reshape(h, w, c))
print("saved logs/board.png")
