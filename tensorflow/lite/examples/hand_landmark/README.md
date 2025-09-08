# build hand_landmark and run on android
### build hand_landmark
Ref:  
https://ai.google.dev/edge/litert/build/android

```
git clone https://github.com/vewe-richard/tensorflow.git
mkdir docker
cp ./tensorflow/tensorflow/lite/tools/tflite-android.Dockerfile docker/
cd docker
docker build . -t tflite-builder -f tflite-android.Dockerfile
cd ../tensorflow
docker run -it -v $PWD:/host_dir tflite-builder bash

# below commands are in the container
cd /host_dir
sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_API_LEVEL}"
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm64 \
  --define=android_dexmerger_tool=d8_dexmerger \
  --define=android_incremental_dexing_tool=d8_dexbuilder \
  //tensorflow/lite/examples/hand_landmark:hand_landmark_detector 

cp bazel-bin/tensorflow/lite/examples/hand_landmark/hand_landmark_detector ./
```

### Run hand_landmark
```
# In host shell

cd tensorflow
adb push hand_landmark_detector /data/local/tmp
adb push tensorflow/lite/examples/hand_landmark/data/hand_landmark.tflite /data/local/tmp/tensorflow/lite/examples/hand_landmark/data/
adb push tensorflow/lite/examples/hand_landmark/data/hand.bmp /data/local/tmp/tensorflow/lite/examples/hand_landmark/data/

adb shell
cd /data/local/tmp
./hand_landmark_detector
```







# convert.py
1. Read hand.png, resize and print image data in c array format  
This array is copied to hand_landmark_detector.h

2. Print out hand landmarks as reference  
Hand Landmarks (x, y, z):
Point 0: (127.7, 218.5, 0.0)
Point 1: (156.6, 206.4, -11.5)
...

3. Generate hand_with_landmarks.png

To run convert.py
```
python3 -m venv tflite_env
source tflite_env/bin/activate
pip install tensorflow numpy pillow matplotlib
cd tensorflow/lite/examples/hand_landmark/data
python convert.py
```

# hand_landmark_detector.cc
Compile:  
```
bazel build -c opt //tensorflow/lite/examples/hand_landmark:hand_landmark_detector
```

Run:   
```
./bazel-bin/tensorflow/lite/examples/hand_landmark/hand_landmark_detector
```

Output:
```
Hand Landmarks (x, y, z):             
Point 0: (127.734, 218.472, 0.00866227
Point 1: (156.626, 206.355, -11.4939) 
Point 2: (176.649, 179.233, -16.7939) 
Point 3: (191.258, 156.23, -21.5387)  
Point 4: (207.59, 141.512, -27.5175)  
Point 5: (150.101, 134.352, -6.87583) 
Point 6: (156.364, 96.443, -10.4339)  
Point 7: (158.755, 74.325, -14.7072)  
Point 8: (160.517, 55.2264, -18.6938) 
Point 9: (129.048, 131.792, -5.50697) 
Point 10: (125.863, 88.0631, -6.98915)
Point 11: (122.335, 63.8259, -11.1839)
Point 12: (119.814, 43.0236, -14.6635)
Point 13: (111.498, 138.004, -6.02488)
Point 14: (104.744, 98.74, -9.50765)  
Point 15: (100.438, 75.8813, -14.816) 
Point 16: (97.8399, 55.2723, -19.0008)
Point 17: (95.2146, 150.759, -7.3783) 
Point 18: (81.6858, 125.009, -11.9655)
Point 19: (72.9781, 109.723, -16.3566)
Point 20: (65.3016, 92.2134, -20.3649)
```



