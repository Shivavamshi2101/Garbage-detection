import argparse
from ultralytics import YOLO

def main(model_path, source, save_result):
    print(f"Loading model: {model_path}")
    # Load the trained model
    model = YOLO(model_path)
    
    print(f"Running inference on source: {source}")
    # Run prediction
    # Removed show=True here so it doesn't close immediately. 
    # We will use result.show() to open it in your default image viewer.
    results = model.predict(source=source, save=save_result, conf=0.25)
    
    # Process results
    for result in results:
        # Show the image with bounded boxes
        result.show()
        
        boxes = result.boxes
        if len(boxes) > 0:
            print(f"\nDetected {len(boxes)} objects:")
            # Print class names and confidences
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                print(f"- {class_name} (Confidence: {conf:.2f})")
        else:
            print("\nNo objects detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO object detection inference.")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the model weights (e.g., best.pt or best.onnx)")
    parser.add_argument("--source", type=str, default=None, help="Source for inference: path to image/video or '0' for webcam")
    parser.add_argument("--save", action="store_true", help="Save the output image/video with bounding boxes")
    
    args = parser.parse_args()
    
    # Prompt for input in the terminal if no source was provided via arguments
    if args.source is None:
        args.source = input("Please enter the path to the image or video (or type 0 for webcam): ").strip()
    
    main(args.model, args.source, args.save)
