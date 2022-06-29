import numpy as np
from scipy.stats import rankdata
from collections import Counter

from time import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam




# Actions that we try to detect
def choose_action():
    '''
    알파벳, 단어 통합 array 반환
    '''
    alphabet = [chr(ord('a') + i) for i in range(26)]
    words = ["angel", "banana", "cry", "dance", "egg", "fun", "game", "house",
             "internet", "jump", "key", "love", "music", "name",
             "open", "paper", "rabbit", "school", "tiger", "video", "walk"]
    total = alphabet + words

    return np.array(total)


# load model by folder name
def build_model():
    '''
    alphabet / word 모델 다르게 사용할 경우 mode 설정
    :return model:
    '''

    actions = choose_action()
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    model.load_weights('action_total_CJ_0626_06.h5')

    return model


def top_n(n, array):
    '''
    유사도 상위 n개 데이터의 인덱스 반환 함수
    :param n:
    :param array:
    :return top_n_idx:
    '''
    ranks = rankdata(array)
    top_n_idx = []
    length = len(array)
    for i in range(length, length-n, -1):
        top_n_idx.append(np.where(ranks == i)[0][0])
    return top_n_idx


def extract_keypoints(results):
    """
    pose(33), left_hand(21), right_hand(21) 의 모든 인덱스를 flatten하는 함수
    ( => 33 * 4 + 21 * 3 + 21 * 3 = 258 )
    :param results:
    :return numpy array (shape=(1, 258)):
    """
    try:
        poses = [{"x": 0, "y": 0, "z": 0, "visibility": 0} for res in range(33)] \
            if 'poseLandmarks' not in results else results['poseLandmarks']
        lhs = [{"x": 0, "y": 0, "z": 0} for res in range(21)] \
            if 'leftHandLandmarks' not in results else results['leftHandLandmarks']
        rhs = [{"x": 0, "y": 0, "z": 0} for res in range(21)] \
            if 'rightHandLandmarks' not in results else results['rightHandLandmarks']
    except Exception as e:
        print(e)
        return
    pose = np.array([[res['x'], res['y'], res['z'], res['visibility']] for res in poses]).flatten()
    lh = np.array([[res['x'], res['y'], res['z']] for res in lhs]).flatten()
    rh = np.array([[res['x'], res['y'], res['z']] for res in rhs]).flatten()

    return np.concatenate([pose, lh, rh])


def result_to_sequence(result):
    '''
    30개의 프레임별 관절 좌표 데이터셋을 numpy array 로 변환하여 이어붙이는 작업
    :param result:
    :return input_sequences:
    '''
    input_sequences = []
    SEQ_LENGTH = 30
    
    for num in range(SEQ_LENGTH):
        keypoint = extract_keypoints(result[num])
        input_sequences.append(keypoint)

    return input_sequences

# prediction generator
class HandSignModel:
    def __init__(self):
        self.actions = choose_action()
        self.model = build_model()

    def predict(self, result):

        '''
        예측 메인 함수
        model build, 입력 데이터 input 형태로 변환, 유사한 알파벳 idx 뽑아서 반환
        :param result:
        :param mode:
        :return top3_alphabet:
        '''
        # 모델 쌓고 나서 prediction 시작
        start = time()

        # 프론트에서 넘어온 시퀀스 30개씩 정리
        sequenceList = []
        for i in range(0, len(result)-30, 4):
            # 50개 중 인덱스가 4로 나뉘어지는 것만으로 자르기(20//4=5개 검사)
            sequence = result_to_sequence(result[i:i+30])
            sequenceList.append(sequence)

        # 시퀀스 별로 나온 예측값 넣기
        resList = []
        for seq in sequenceList:
            res = self.model.predict(np.expand_dims(seq, axis=0))[0]
            resList.append(res)

        afterPredict = time()

        # 최상위 3개 알파벳/단어
        predictedList = []
        for top3 in resList:
            predictedList += top_n(3, top3)

        # 5개의 예측값 리스트 중 최빈값 뽑기
        predictedCounter = Counter(predictedList)
        sortedPredictedList = [i[0] for i in predictedCounter.most_common(n=3)]
        top3_alphabet = [self.actions[i] for i in sortedPredictedList[0:3]]
        
        end = time()
        print('총 걸리는 시간: ' , afterPredict-start, end-start)
        
        return top3_alphabet