from tensorflow import keras
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Mediapipe pose model


def make_model(input_shape, n_classes):
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(n_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def get_processed_pose(img, results, draw, W, H):
    def extract_xys(ob):
        p = np.array([ob.x, ob.y])
        return p

    two_d_pose = {"j2": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                  "j5": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                  "j8": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
                  "j11": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                  "j9": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
                  "j12": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
                  "j10": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
                  "j13": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
                  "j0": results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE],
                  "j3": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
                  "j6": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
                  "j4": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST],
                  "j7": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST],
                  "j16": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE],
                  "j17": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
                  }

    pose_d = dict(map(lambda x: (x[0], extract_xys(x[1])), two_d_pose.items()))

    pose = np.array([
        pose_d["j0"],
        pose_d["j2"],
        pose_d["j3"],
        pose_d["j4"],
        pose_d["j5"],
        pose_d["j6"],
        pose_d["j7"],
        pose_d["j8"],
        pose_d["j9"],
        pose_d["j10"],
        pose_d["j11"],
        pose_d["j12"],
        pose_d["j13"],
        pose_d["j16"],
        pose_d["j17"]]
    )
    pose = pose * np.array([W, H])  # Get original coordinates
    f_pose = np.reshape(pose, (30))

    if draw:
        img.flags.writeable = True
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    return f_pose, img
