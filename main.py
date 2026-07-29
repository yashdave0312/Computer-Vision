import cv2
import numpy as np

vid = cv2.VideoCapture(0)

try:
    while True:

        ret, frame = vid.read()

        if not ret:
            print("Failed to capture frame. Check webcam connection.")
            break


        # Get frame dimensions
        height, width, _ = frame.shape


        # Create detection box (ROI)
        x1 = width // 3
        y1 = height // 3

        x2 = width * 2 // 3
        y2 = height * 2 // 3

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        roi = frame[y1:y2, x1:x2]


        # Display frame
        cv2.imshow("RGB Color Detection", frame)


        # Separate RGB channels from ROI
        b = roi[:, :, 0]
        g = roi[:, :, 1]
        r = roi[:, :, 2]


        # Calculate mean RGB values
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)


        # Detect dominant colour
        if b_mean > g_mean and b_mean > r_mean:
            color = "Blue"

        elif g_mean > r_mean and g_mean > b_mean:
            color = "Green"

        else:
            color = "Red"


    
        cv2.putText(
            frame,
            f"Color: {color}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        # Display RGB values
        cv2.putText(
            frame,
            f"RGB: {int(r_mean)}, {int(g_mean)}, {int(b_mean)}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except Exception as e:
    print("Error occurred:", e)


finally:
    vid.release()
    cv2.destroyAllWindows()
    