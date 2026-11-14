import time
import cv2
import rerun as rr

def main():
    rr.init("Real-Time Camera + Zero-DCE", spawn=True)

    # Abrir la cámara (0 suele ser la webcam integrada; prueba 1 o 2 para cámaras USB/RealSense)
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Verifica el índice o los permisos del dispositivo.")

    print("Transmitiendo video a Rerun. Presiona Ctrl+C en la terminal para detener.")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el fotograma.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("time_sec", duration=time.time())

            rr.log("camera/image", rr.Image(frame_rgb).compress(jpeg_quality=85))

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nCaptura detenida por el usuario.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()