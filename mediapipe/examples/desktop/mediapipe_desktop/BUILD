licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

cc_binary(
    name = "mediapipe_desktop",
    srcs = [
        "mediapipe_desktop.cc",
        "mediapipe_desktop.h",
    ],
    linkshared = True,
    deps = [
        "//mediapipe/examples/desktop/face_mesh_sdk",
        "//mediapipe/examples/desktop/hair_segmentation_sdk",
        "//mediapipe/examples/desktop/hand_tracking_sdk",
    ],
)
