#%%
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from tqdm.notebook import tqdm
from copy import deepcopy
from datetime import datetime as dt
from sklearn.model_selection import train_test_split

# fm.get_fontconfig_fonts()
# matplotlib.rc('font', family=fm.FontProperties(fname='C:/Windows/Fonts/NanumSquareR.ttf').get_name())


#%%
# 신호 기록 가져오기
with open('sample_20200601_pointfinger.txt', 'r') as openfile :
    samples = openfile.readlines()

tmp_timests = [ samples[i][:-1] for i in range(len(samples)) if i%3==0 ]
tmp_samples = [ samples[i][:-1] for i in range(len(samples)) if i%3==1 ]
print("We get times and sample : {}, {}".format(len(tmp_timests), len(tmp_samples)))

#%%
# 플롭 해보기
def plot(t, s, title='', xlabel='Time (sec)', ylabel='Magnitude', dup=False, style='-', ylim=None, **kwargs) :
    plt.figure(figsize=(18, 5))
    plt.plot(t, s, style, ms=4, lw=1, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim :
        plt.yticks(np.linspace(ylim[0], ylim[1], 10)) 
        plt.ylim(ylim)
    plt.title(title, fontsize=15, pad=20)
    if not dup : 
        plt.show()

#%%
# 다시 시간 순으로 샘플링해야됨
def getTimeMS(timestamp) : 
    tmp = timestamp.replace('2020-06-01 09:', '')
    return int(tmp[:2])*60 + float(tmp[3:])

def sampling2Time(tmp_timests, tmp_samples) :
    timests, samples = list(), list()
    set_timests = sorted(list(set(tmp_timests)))
    pbar = tqdm(total=len(set_timests))
    while len(set_timests) >= 2 :
        sinx, einx = tmp_timests.index(set_timests[0]), tmp_timests.index(set_timests[1])
        st, et = getTimeMS(set_timests[0]), getTimeMS(set_timests[1])
        timests.extend(np.linspace(st, et, einx-sinx+1)[:-1])
        samples.extend(tmp_samples[sinx:einx])
        pbar.update(1)
        del set_timests[0]
    pbar.close()

    if len(timests) != len(samples) :
        print("The lengths of timestamps and samples are different! ({} vs {})".format(len(timests, samples)))
    else :
        timests, samples = np.array(timests)-np.min(timests), np.array(samples, dtype=np.float)

        return timests, samples

timests, samples = sampling2Time(tmp_timests, tmp_samples)
# plot(timests, samples)

# 초기화 시간 제거하기
init_end_inx = 0
init_end_val = 175
while True :
    if samples[init_end_inx] < init_end_val :
        break
    init_end_inx += 1
timests, samples = timests[init_end_inx:], samples[init_end_inx:]
timests -= np.min(timests)
# plot(timests, samples)

#%% 
def getIdx4Span(timests, span) :
    indices, span = list(), sorted(span)
    for i in range(len(timests)) :
        if timests[i] > span[0] :
            indices.append(i)
            del span[0]
            if len(span) == 0 :
                return indices
            else :
                continue
    return indices + [len(timests)-1]

#%%
def getWeightedSamples(subsamples, subtimests, ts=0.05, ratio=0.1) :
    stime, si, i, mean, s, cnt = subtimests[0], 0, 0, 0, 0, 0
    subsamples = deepcopy(subsamples)
    while i < len(subtimests) :
        if subtimests[i] - stime < ts :
            s += subsamples[i]
            cnt += 1
        elif cnt > 0 :
            if mean == 0 :
                mean = s/cnt
            else :
                mean = mean*(1-ratio)+s/cnt*ratio
            subsamples[si:i] -= mean
            stime = subtimests[i]
            s, cnt, si = 0, 0, i
        else :
            print(stime, si, i, mean, s, cnt)
        i += 1
    if i - si > 0 :
        subsamples[si:i] -= mean
    return subsamples

# 중간 구간에 바이어스 하기
test_span = [48.5, 60]
test_idxs = getIdx4Span(timests, test_span)
test_samples = samples[test_idxs[0]:test_idxs[1]]
bias_hype_value = 173.5
bias_hype_timests = timests[test_idxs[0]:test_idxs[1]]
bias_hype_samples = np.abs(test_samples-bias_hype_value)

# plot(bias_hype_timests, test_samples)
# plot(bias_hype_timests, bias_hype_samples, style='-', ylim=[0, 14], color='green')
cases, nsmpl = 10, 20
total, cnt = 0, 0
for i in range(cases) :
    stime = dt.now()
    p = 0.7
    T = 0.01
    bias_wegh_samples = getWeightedSamples(samples, timests, T, p)
    bias_wegh_samples = np.abs(bias_wegh_samples)
    delay = dt.now() - stime
    delay = (delay.seconds*1000) + (delay.microseconds*0.001)
    print("Total time(ms) : {}, Size : {}, Processing Speed(ms/sample) : {}".format(
        delay, len(samples), delay/len(samples)
    ))
    total, cnt = total+delay, cnt+len(samples)
timests_length = timests[-1]-timests[0]
print(timests_length)
print("@ Finally 10 cases average - Total time(ms) : {}, Size : {}, Processing Speed(ms/sample) : {}".format(
    total, cnt, (total/10)/(timests_length/0.01)
))


# plot(bias_hype_timests, bias_wegh_samples[test_idxs[0]:test_idxs[1]], style='-', ylim=[0, 14], color='red')



#%%
# T 간격으로 다시 샘플링하기
pbar = tqdm(total=len(timests))

t, stime, T, i = timests[0], timests[0], 0.01, 0
resamples, retimests = list(), list()
total, cnt = 0, 0
while i < len(timests) :
    if timests[i] < stime+T :
        total += bias_wegh_samples[i]
        cnt += 1
    else :
        resamples.append(total/cnt)
        retimests.append(stime)
        total, cnt, stime = bias_wegh_samples[i], 1, timests[i]
    pbar.update(1)
    i += 1
pbar.close()
resamples, retimests = np.array(resamples),  np.array(retimests)      
plot(retimests, resamples, style='-', color='black')

#%%
total, cnt = 0, 0
for i in range(cases) :
    stime = dt.now()
    
    n, b, s = 1, 10, 0.9
    grad_samples, filt_samples = deepcopy(resamples), deepcopy(resamples)
    grad_samples[:-1] -= grad_samples[1:]
    filt_samples[grad_samples<n] = 0
    filt_samples[filt_samples>b] = b
    nois_timests = retimests
    nois_samples = filt_samples

    smoo_samples = nois_samples
    for i in range(3) :
        i = 1
        while i < len(smoo_samples) :
            smoo_samples[i] = smoo_samples[i]+(smoo_samples[i-1]-smoo_samples[i])*s
            i+=1
    smoo_samples[smoo_samples>1] = 1

    delay = dt.now() - stime
    delay = (delay.seconds*1000) + (delay.microseconds*0.001)
    print("Total time(ms) : {}, Size : {}, Processing Speed(ms/20 sample) : {}".format(
        delay, len(samples), delay/len(samples)*nsmpl
    ))
    total, cnt = total+delay, cnt+len(samples)
print("@ Finally 10 cases average - Total time(ms) : {}, Size : {}, Processing Speed(ms/20 sample) : {}".format(
    total, cnt, total/cnt*nsmpl
))

# plot(nois_timests, nois_samples, style='-',color='black')

# plot(nois_timests, smoo_samples, style='o', color='black')


#%%
# 훈련 데이터 만들기
def make_datasets(nois_timests, smoo_samples, d, T) :
    if (d*1000) % (T*1000) != 0 : return print("d {} and T {} are not match!")
    else :
        datasets, i = {'time':[], 'smpl':[]}, 0
        while i < len(smoo_samples) :
            while i < len(smoo_samples) and smoo_samples[i] < 0.05  :
                i += 1
                continue
            si = i
            while i < len(smoo_samples) and smoo_samples[i] > 0.05 :
                i += 1
                continue 
            ei = i
            if np.mean(smoo_samples[si:ei]) > 0.15 :
                datasets['time'].append(nois_timests[si:ei])
                datasets['smpl'].append(smoo_samples[si:ei])
            i += 1
        return datasets

def labeling(time, smpl, n, dim) :
    # plot(time, smpl, '-', color='black')
    
    indices = np.linspace(0, 19.99999, len(time)+dim)
    smpl = np.insert(smpl, 0, np.random.uniform(0, 0.005, dim))
    smpl = np.insert(smpl, len(smpl), np.random.uniform(0, 0.005, dim))
    chunks = [
        (np.insert(smpl[i:i+dim], 0, np.array([int(indices[i]==j) for j in range(dim)])), int(indices[i])) 
        for i in range(len(time)+dim)
    ]

    ## Plot!
    # nrows = 5
    # fig, ax = plt.subplots(nrows, len(chunks)//nrows + (0 if len(chunks)%nrows==0 else 1), figsize=(20, 10))
    # for i in range(len(ax)) :
    #     for j in range(len(ax[i])) :
    #         ax[i][j].set_xticks([])
    #         ax[i][j].set_yticks([])
    #         ax[i][j].set_ylim([0, 1])
    # for i, chunk in enumerate(chunks) :
    #     ax[i%nrows][i//nrows].plot(range(len(chunk[0])), chunk[0], '-', color='black')
    #     ax[i%nrows][i//nrows].text(8, 0.8, str(chunk[1]), fontsize=15)
    # plt.show()
    
    return chunks

d, n, dim = 0.1, 10, 20
datasets = make_datasets(nois_timests, smoo_samples, d, T)

total_datasets = list()
for i in range(len(datasets['time'])) :
    total_datasets += labeling(datasets['time'][i], datasets['smpl'][i], n=n, dim=dim)

inputs, labels = list(), list()
for total_dataset in total_datasets :
    inputs.append(total_dataset[0])
    labels.append(total_dataset[1])
inputs, labels = np.array(inputs), np.array(labels)

print("Train set length : {}, Vector length : {}".format(len(total_datasets), len(total_datasets[0][0])))

#%%
# 섞어서 훈련과 테스트셋 만들기
data_train, data_test, labl_train, labl_test = train_test_split(inputs, labels, test_size=0.25, shuffle=True, random_state=42)

print("We make {} length of trains, {} length of test".format(len(data_train), len(data_test)))

#%%
print(len(data_train)/18*130)
print(len(data_test)/18*130)

#%%
### 신경망 학습을 위한 모듈 가져오기
import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential

#%%
# 모델 만들기
model = Sequential()
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(dim, activation='softmax'))
model.build(input_shape=(None, 40))
model.summary()

#%%
# 훈련 시작
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(data_train, labl_train, epochs=5000)

#%%
# 모델 평가
test_loss, test_acc = model.evaluate(data_test, labl_test, verbose=2)

#%%
# Predictation
stime = dt.now()
pbar = tqdm(total=len(data_test))

predicts = list()
for i in range(len(data_test)) :    
    p = np.argmax(model.predict(data_test[i:i+1]).tolist()[0])
    predicts.append(p)
    pbar.update(1)

delay = dt.now() - stime
pbar.close()
delay = (delay.seconds*1000) + (delay.microseconds*0.001)
print("Total time(ms) : {}, Size : {}, Processing Speed(ms/20 sample) : {}".format(
    delay, len(data_test), delay/len(data_test)
))

#%%
plot(range(len(predicts)), predicts, xlabel="Inputs", ylabel="Finger Positions", title="Predictations")
plot(range(len(predicts)), labels, xlabel="Inputs", ylabel="Finger Positions", title="Labels")

#%%
model.save('./my_model.h5')
#%%
print(trains_inputs[0:1])
#%%
# len(pvalues)

# #%%
# # 스펙트럼 관찰
# import scipy.signal

# f, P = scipy.signal.periodogram(np.array(samples), int(1/(timests[1]-timests[0])), nfft=len(samples))

# plt.subplot(211)
# plt.plot(f, P)
# plt.title("선형 스케일")

# plt.subplot(212)
# plt.semilogy(f, P)
# plt.title("로그 스케일")

# plt.tight_layout()
# plt.show()
