# Installation

 - pytorch (conda)
 - tensorrt (pip)
 - lapx (pip)

Run this command to get the models:

```
yolo export model=yolov8m-pose.pt format=engine
```
Copy the generated model to the `models/` folder.
Run with sudo on linux (depends on the usb device, arduino normally `/dev/ttyACM0`)
