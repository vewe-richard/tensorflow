cc_binary(
    name = "hand_landmark_detector",
    srcs = ["hand_landmark_detector.cc","hand_landmark_detector.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/examples/label_image:bitmap_helpers",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite:minimal_logging",
    ],
    copts = ["-Wno-sign-compare"],
)

