import os
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
TRAINED_WEIGHTS_PATH = "runs/segment/robotic_grasp/yolov8_grasp_items_seg_version_2/weights/best.pt"

# Fallback to yolov8n-seg if the custom weights are not found on your current path
if not os.path.exists(TRAINED_WEIGHTS_PATH):
    print(f"[WARN] Custom weights not found at '{TRAINED_WEIGHTS_PATH}'. Falling back to default 'yolov8n-seg.pt' for testing.")
    TRAINED_WEIGHTS_PATH = "yolov8n-seg.pt"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30
# =======================================================


def main():
    print(f"[MODEL] Loading YOLOv8 segmentation model from: {TRAINED_WEIGHTS_PATH}")
    model = YOLO(TRAINED_WEIGHTS_PATH)

    print("[CAMERA] Starting Intel RealSense D435i stream...")
    pipeline = rs.pipeline()
    config = rs.config()

    # 1. Enable BOTH color and depth streams with matching dimensions and frame rate
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FRAME_FPS)
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FRAME_FPS)
    
    # Start the camera pipeline with our configuration
    pipeline.start(config)

    # 2. Setup Alignment & Colorization Utility
    # We align the depth frame to match the color frame's coordinate perspective
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # This colorizer converts raw 16-bit depth values into a beautiful 8-bit color map automatically
    colorizer = rs.colorizer()

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    print("[SYSTEM] Perception loop running. Focus the OpenCV window and press 'q' to shut down.")
    
    try:
        while True:
            # Grab synchronized frame sets from the camera pipeline
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to the color frame space
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            # Verify both frames arrived successfully before doing any processing
            if not color_frame or not depth_frame:
                continue

            # Convert color and depth frames into standard numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Colorize the 16-bit depth image so it renders correctly in OpenCV
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(depth_color_frame.get_data())
            
            overlay_canvas = color_image.copy()

            # Run segmentation model inference
            results = model.predict(source=color_image, save=False, verbose=False)
            result = results[0]

            if result.masks is not None and len(result.masks.data) > 0:
                masks_numpy = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                class_names = model.names

                for i in range(len(masks_numpy)):
                    class_id = classes[i]
                    class_name = class_names[class_id]
                    confidence = confidences[i]
                    color = colors[class_id].tolist()

                    single_mask = masks_numpy[i]
                    resized_mask = cv2.resize(single_mask, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255

                    # Blend the color-coded mask overlay on top of our original frame
                    color_mask_img = np.zeros_like(color_image)
                    color_mask_img[binary_mask > 0] = color
                    overlay_canvas[binary_mask > 0] = cv2.addWeighted(
                        overlay_canvas, 0.4, color_mask_img, 0.6, 0
                    )[binary_mask > 0]

                    # Extract outer contours and draw boundary lines
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_canvas, contours, -1, color, 2)

                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        label_text = f"{class_name} ({confidence*100:.0f}%)"

                        cv2.putText(
                            overlay_canvas, 
                            label_text, 
                            (x, max(y - 10, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )

            # Display streams in separate OpenCV windows
            cv2.imshow("Segmentation Result", overlay_canvas)
            cv2.imshow("Depth Image", depth_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("[CAMERA] Shutting down streams and cleaning up system environment...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[SYSTEM] Exit completed cleanly.")

if __name__ == "__main__":
    main()