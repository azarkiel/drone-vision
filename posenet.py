import heapq
import tensorflow as tf
import numpy as np
import cv2

keypoint_decoder = [
    "nose",             # 0
    "leftEye",          # 1
    "rightEye",         # 2
    "leftEar",          # 3
    "rightEar",         # 4
    "leftShoulder",     # 5
    "rightShoulder",    # 6
    "leftElbow",        # 7
    "rightElbow",       # 8
    "leftWrist",        # 9
    "rightWrist",       # 10
    "leftHip",          # 11
    "rightHip",         # 12
    "leftKnee",         # 13
    "rightKnee",        # 14
    "leftAnkle",        # 15
    "rightAnkle",       # 16
]

keypoint_encoder = {x: i for i,x in enumerate(keypoint_decoder)}

# Pairs represents the lines connected from joints
# e.g. (5,6) is from leftShoulder to rightShoulder
# https://www.tensorflow.org/lite/models/pose_estimation/overview
keypoint_lines = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]
face_keypoints = [0, 1, 2, 3, 4]

# define the skeleton. code from Google's tfjs-models
# each tuple is (parent, child)
poseChain = [
  ('nose',          'leftEye'), 
  ('leftEye',       'leftEar'), 
  ('nose',          'rightEye'),
  ('rightEye',      'rightEar'), 
  ('nose',          'leftShoulder'),
  ('leftShoulder',  'leftElbow'), 
  ('leftElbow',     'leftWrist'),
  ('leftShoulder',  'leftHip'), 
  ('leftHip',       'leftKnee'),
  ('leftKnee',      'leftAnkle'), 
  ('nose',          'rightShoulder'),
  ('rightShoulder', 'rightElbow'), 
  ('rightElbow',    'rightWrist'),
  ('rightShoulder', 'rightHip'), 
  ('rightHip',      'rightKnee'),
  ('rightKnee',     'rightAnkle')
]
parentChildrenTuples = [(keypoint_encoder[parent], keypoint_encoder[child]) for (parent, child) in poseChain]
parentToChildEdges = [childId for (_, childId) in parentChildrenTuples]
childToParentEdges = [parentId for (parentId, _) in parentChildrenTuples]

class PoseNet():
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.model_fn = self.model.signatures["serving_default"]
        self.OUTPUT_STRIDE = 16
        self.INPUT_WIDTH = self.INPUT_HEIGHT = 224
        
    def prepare_input(self, img):
        ''' img is a (height, width, 3) image. this will resize the image to the PoseNet input dimensions, and add a batch dimension. Return an image with shape (1, INPUT_HEIGHT, INPUT_WIDTH, 3). Original image is not modified. '''
        img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        img_copy = img_copy - [123.15, 115.90, 103.06]
        img_copy = cv2.resize(img_copy, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        img_copy = np.expand_dims(img_copy, axis=0)
        img_copy = tf.convert_to_tensor(img_copy, dtype=tf.float32)
        return img_copy

    def predict(self, img):
        ''' invoke the TensorFlow Lite model. Return heatmaps, offsets, displacementFoward, and displacementBackward tensors '''
        img_copy = img

        output = self.model_fn(img_copy)
        return output.values()

    def predict_singlepose(self, img):
        ''' Wrapper around decode_singlepose. Return a list of 17 keypoints '''
        img_input = self.prepare_input(img)
        heatmaps, offsets, _, _ = self.predict(img_input)

        heatmaps = np.squeeze(heatmaps)
        offsets = np.squeeze(offsets)

        keypoints = decode_singlepose(heatmaps, offsets, self.OUTPUT_STRIDE)
        
        scaleY = img.shape[0] / self.INPUT_HEIGHT
        scaleX = img.shape[1] / self.INPUT_WIDTH
        scale = np.array([scaleX, scaleY])
        for keypoint in keypoints:
            keypoint['position'] = np.round(keypoint['position'] * scale).astype(int)

        return keypoints

def decode_singlepose(heatmaps, offsets, outputStride):
    ''' Decode heatmaps and offets output to keypoints. Return a list of keypoints, each keypoint is a dictionary, with keys 'pos' for position np.ndarray([x,y]) and 'score' for confidence score '''
    numKeypoints = heatmaps.shape[-1]

    def get_keypoint(i):
        sub_heatmap = heatmaps[:,:,i]    # heatmap corresponding to keypoint i
        y, x = np.unravel_index(np.argmax(sub_heatmap), sub_heatmap.shape)    # y, x position of the max value in heatmap
        score = sub_heatmap[y,x]    # max value in heatmap

        # convert x, y to coordinates on the input image
        y_image = y*outputStride + offsets[y, x, i]
        x_image = x*outputStride + offsets[y, x, i + numKeypoints]
        
        # position is wrapped in a np array to support vector operations
        pos = np.array([x_image, y_image])
        return {'position': pos, 'score': score}

    keypoints = [get_keypoint(i) for i in range(numKeypoints)]
    
    return keypoints

# code from https://github.com/tensorflow/tfjs-models/tree/master/posenet
def decode_multipose(heatmaps, offsets, displacementFwd, displacementBwd, outputStride, maxPose, threshold=0.5, localMaxR=20, NmsRadius=20):
    height, width, numKeypoints = heatmaps.shape
    sqNmsRadius = NmsRadius**2
    poses = []
    # use a max heap here
    queue = []

    for keypointId in range(numKeypoints):
        sub_heatmap = heatmaps[:,:,keypointId]
        # only consider points with score >= threshold as root candidate
        candidates = np.argwhere(sub_heatmap >= threshold)

        for candidate in candidates:
            y, x = candidate
            # check if the candidate is local maximum in the local window
            # [(x-localMaxR, y-localMaxR), (x+localMaxR, y+localMaxR)]
            x0 = max(0, x-localMaxR)
            x1 = min(width, x+localMaxR+1)
            y0 = max(0, y-localMaxR)
            y1 = min(height, y+localMaxR+1)
            local_window = sub_heatmap[y0:y1, x0:x1]
    
            max_score = np.max(local_window)
            if sub_heatmap[y,x] == max_score:
                queue.append((
                    sub_heatmap[y,x],   # score
                    (y,x),              # position
                    keypointId,         # keypoint id            
                ))
    heapq._heapify_max(queue)

    # generate at most maxPose object instances
    while len(poses) < maxPose and len(queue) > 0:
        # root will be the keypoint with highest score from the queu
        root = heapq._heappop_max(queue)
        score, (y,x), keypointId = root

        # calculate position of keypoint on original input image
        # offsets has shape (width, height, 2*numKeypoints)
        y_image = y*outputStride + offsets[y, x, keypointId]
        x_image = x*outputStride + offsets[y, x, keypointId+numKeypoints]

        # reject root if it is within a disk of nmsRadius from the corresponding part of a previously detected instance
        reject = False
        for pose in poses:
            y_pose, x_pose = pose[keypointId]
            if (y_pose - y_image)**2 + (x_pose - x_image)**2 <= sqNmsRadius:
                reject = True
                break
        if reject:
            continue

        instanceKeypoints = [0] * numKeypoints
        instanceKeypoints[keypointId] = (score, (y,x))

        numEdges = len(parentToChildEdges)
        for edge in range(numEdges-1, -1, -1):  # from numEdges-1 to 0 inclusive
            sourceKeypointId = parentToChildEdges[edge]
            targetKeypointId = childToParentEdges[edge]
            if instanceKeypoints[sourceKeypointId] and not instanceKeypoints[targetKeypointId]:
                instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint

        poses.append({
            keypoints: None, # get keypoints from decodePose()
            score: None, # get instance score from the instance
        })
    return queue

def traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scores, offsets, outputStride, displacements, offsetRefineStep=2):
    y,x = sourceKeypoint
    sourceKeypointIndices = (y/outputStride, x/outputStride)
    displacement = (displacements[y,x,edgeId], displacements[y,x,edgeId])

    displacedPoint = (y+displacement[0], x+displacement[1])
    targetKeyPoint = displacedPoint
    for i in range(offsetRefineStep):
        i = 0


def draw_keypoints(img, keypoints, threshold=0.5):
    ''' Draw keypoints on the given image '''
    for i, keypoint in enumerate(keypoints):
        pos = keypoint['position']
        score = keypoint['score']
        if score < threshold:
            continue    # skip if score is below threshold

        cv2.circle(img,tuple(pos),5,(0,255,0),-1)    # draw keypoint as circle
        keypoint_name = keypoint_decoder[i]
        cv2.putText(img,keypoint_name,tuple(pos),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) # put the name of keypoint

    return img

def draw_pose(img, keypoints, threshold=0.2, color=(0,255,0), keypointRadius=5, keypointThickness=-1, lineThickness=2):
    ''' Draw pose on img. keypoints is a list of 17 keypoints '''

    # draw keypoints of the face (eyes, ears and nose)
    for keypointId in face_keypoints:
        pos = keypoints[keypointId]['position'].astype(int)
        score = keypoints[keypointId]['score']
        if score < threshold:
            continue

        cv2.circle(img, tuple(pos), keypointRadius, color, keypointThickness)

    # draw lines connecting joints
    for (id1, id2) in keypoint_lines:
        pos1 = keypoints[id1]['position']
        pos2 = keypoints[id2]['position']
        score1 = keypoints[id1]['score']
        score2 = keypoints[id2]['score']

        if score1 < threshold or score2 < threshold:
            continue

        cv2.line(img, tuple(pos1), tuple(pos2), color, lineThickness)
    
    return img

def detect_pose(keypoints, threshold=0.1):
    result = {}

    # t-pose
    result['t-pose'] = True
    tpose_series = ['leftWrist', 'leftElbow', 'leftShoulder', 'rightShoulder', 'rightElbow', 'rightWrist']  # consider these 5 points
    tpose_series = [keypoint_encoder[x] for x in tpose_series]                                              # convert to keypoint id
    tpose_series = [keypoints[x] for x in tpose_series]                                                     # obtain positions from keypoints

    reject = False
    for tpose_point in tpose_series:
        if tpose_point['score'] < 0.2:
            reject = True
            break
    if reject:
        result['t-pose'] = False
    else:
        for i in range(len(tpose_series)-1):
            vector = tpose_series[i+1]['position'] - tpose_series[i]['position']  # get vector of consecutive keypoints
            cosAngle2 = vector[1]**2 / vector.dot(vector)       # calculate cos angle squared wrt to vertical

            if cosAngle2 > threshold:
                result['t-pose'] = False
                break

    # left-hand-up and right-hand-up
    for side in ['left', 'right']:
        key = f'{side}-hand-up'
        result[key] = True
        handup_series = ['Wrist', 'Elbow', 'Shoulder']                     # consider these 3 points
        handup_series = [keypoint_encoder[f'{side}{x}'] for x in handup_series]   # convert to keypoint id
        handup_series = [keypoints[x] for x in handup_series]                  # obtain positions

        reject = False
        for handup_point in handup_series:
            if handup_point['score'] < 0.2:
                reject = True
                break
        if reject:
            result[key] = False
        else:
            for i in range(len(handup_series)-1):
                vector = handup_series[i+1]['position'] - handup_series[i]['position']  # get vector
                if vector[1] < 0:
                    result[key] = False
                    break
                
                cosAngle = vector[0] / np.linalg.norm(vector)   # calculate cos angle wrt to horizontal

                if cosAngle > threshold:
                    result[key] = False
                    break

    return result

if __name__ == '__main__':
    # set up posenet
    model_path = 'posenet_resnet50float_stride16'
    posenet = PoseNet(model_path)

    # read image and prepare input to shape (1, height, width, 3)
    # img = cv2.imread('person.jpg')
    cap = cv2.VideoCapture(0) 
    while True:        
        success, img = cap.read()   # read webcam capture

        # apply model
        keypoints = posenet.predict_singlepose(img)
        # draw keypoints on original image
        draw_keypoints(img, keypoints)
        draw_pose(img, keypoints)
        poses = detect_pose(keypoints)

        cv2.imshow('posenet', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # terminate window when press q
            break
    cv2.destroyAllWindows()