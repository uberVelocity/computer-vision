import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Set new resolution
cap.set(3, 320)
cap.set(4, 480)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() # returns true/false depending whether frame is read correctly

    # Print resolution in terminal    
    print(cap.get(3))
    print(cap.get(4))

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()