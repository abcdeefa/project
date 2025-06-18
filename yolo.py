import cv2
from ultralytics import YOLO

def initialize_camera(width=640, height=480):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def run_detection(model_path='best.pt', conf_threshold=0.5):
    print("YOLO 모델 로딩")
    model = YOLO(model_path)

    print("웹캠 연결")
    cap = initialize_camera()

    print("Q키 누르면 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        results = model(frame, conf=conf_threshold)
        annotated = results[0].plot()

        cv2.imshow("YOLOv8 객체 탐지", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()
