import random
import math
import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class VectorDataset(Dataset):
    def __init__(self, X, S, Y, T=None, D=None):
        self.X = X
        self.S = S
        self.Y = Y
        self.T = T
        self.D = D
    
    def __getitem__(self, i):
        x, s, y = self.X[i], self.S[i], self.Y[i]
        if self.T != None:
            t = self.T[i]
            return x, t, s, y
        else:
            return x, s, y
    
    def __len__(self):
        return self.X.shape[0]

def setSeed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def getOneHot(df, cols):
    T = df[cols].values
    D = df[cols].nunique().values
    X = []
    for col in cols:
        X.append(pd.get_dummies(df[col]).values)
    X = np.concatenate(X, axis=1)
    return X, T, D

def getBinary(df, cols):
    labels = df[cols].apply(lambda s: np.median(s)).values
    x = df[cols].values
    xs = np.zeros_like(x)
    for j in range(len(labels)):
        if x[:,j].max() == labels[j]:
            xs[:,j] = x[:,j]
        else:
            xs[:,j] = (x[:,j] > labels[j]).astype(int)
    df = pd.DataFrame(xs, columns=cols)
    return df

def loadMnist(color=True):
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Scale(64),\
        transforms.CenterCrop(64),transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    if not color:
        n = len(train_data)
        X_train = torch.zeros(n, 1, 64, 64)
        Y_train = train_data.targets
        S_train = Y_train
        for i in range(n):
            X_train[i,0] = transform(train_data.data[i]).squeeze()
        X_train /= X_train.max()
        
        n = len(test_data)
        X_test = torch.zeros(n, 1, 64, 64)
        Y_test = test_data.targets
        S_test = Y_train
        for i in range(n):
            X_test[i,0] = transform(test_data.data[i]).squeeze()
        X_test /= X_test.max()
    else:
        # train
        n = len(train_data)
        X_train = torch.zeros(n, 3, 64, 64)
        S_train = torch.arange(n) % 3
        Y_train = train_data.targets
        for i in range(n):
            X_train[i,S_train[i]] = transform(train_data.data[i]).squeeze()
        X_train /= X_train.max()
        
        # test
        n = len(test_data)
        X_test = torch.zeros(n, 3, 64, 64)
        S_test = torch.arange(n) % 3
        Y_test = test_data.targets
        for i in range(n):
            X_test[i,S_test[i]] = transform(test_data.data[i]).squeeze()
        X_test /= X_test.max()
    
    train_data = VectorDataset(X_train, S_train, Y_train)
    test_data = VectorDataset(X_test, S_test, Y_test)
    
    return train_data, test_data

def loadAdult(discrete=True):
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country', 'salary']

    df_train = pd.read_csv('data/adult/adult.data', names=cols)
    df_test = pd.read_csv('data/adult/adult.test', names=cols, skiprows=1)

    num_train = df_train.shape[0]
    num_test = df_test.shape[0]
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    print('train_size {}, test_size {}'.format(num_train, num_test))
    
    category_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    continuous_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    feat_cols = category_cols + continuous_cols

    S = (df['sex'] == 'Male').values.astype(int)
    Y = (df['salary'].apply(lambda x: x == '<=50K' or x == '<=50K.')).values.astype(int)
    del df['sex']
    del df['salary']

    S = torch.LongTensor(S)
    Y = torch.LongTensor(Y)
    S_train, S_test = S[:num_train], S[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    df['age'] = pd.cut(df['age'], bins=8, labels=False)
    df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=8, labels=False)
    df['fnlwgt'] = pd.cut(np.log(df['fnlwgt']), bins=8, labels=False)
    df[['capital-gain', 'capital-loss']] = getBinary(df, ['capital-gain', 'capital-loss'])
    df[feat_cols] = df[feat_cols].apply(lambda col: LabelEncoder().fit_transform(col))
    
    if discrete:
        X, T, D = getOneHot(df, feat_cols)
        X = torch.FloatTensor(X)
        T = torch.LongTensor(T)
        D = list(D)

        X_train, X_test = X[:num_train], X[num_train:]
        T_train, T_test = T[:num_train], T[num_train:]
        
        train_data = VectorDataset(X_train, S_train, Y_train, T_train)
        test_data = VectorDataset(X_test, S_test, Y_test, T_test)
        train_data.D = D

        return train_data, test_data
    else:
        df[feat_cols] = MinMaxScaler().fit_transform(df[feat_cols])
        X = df[feat_cols].values
        X = torch.FloatTensor(X)
        
        X_train, X_test = X[:num_train], X[num_train:]

        train_data = VectorDataset(X_train, S_train, Y_train)
        test_data = VectorDataset(X_test, S_test, Y_test)
        
        return train_data, test_data

def loadCompas(discrete=True):
    df = pd.read_csv('data/compas.csv')
    df = shuffle(df)
    num_train = int(0.8*df.shape[0])
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))
    
    df['Number_of_Priors'] = pd.cut(df['Number_of_Priors'],bins=8,labels=False)
    
    S = df['African_American'].values
    Y = df['Two_yr_Recidivism'].values
    S = torch.LongTensor(S)
    Y = torch.LongTensor(Y)
    S_train, S_test = S[:num_train], S[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]
    df = df.drop(['African_American', 'Two_yr_Recidivism'], axis=1)
    
    if discrete:
        X, T, D = getOneHot(df, df.columns)
        X = torch.FloatTensor(X)
        T = torch.LongTensor(T)
        D = list(D)
        
        X_train, X_test = X[:num_train], X[num_train:]
        T_train, T_test = T[:num_train], T[num_train:]
        
        train_data = VectorDataset(X_train, S_train, Y_train, T_train)
        test_data = VectorDataset(X_test, S_test, Y_test, T_test)
        train_data.D = D

        return train_data, test_data
    else:
        X = df.values
        X = MinMaxScaler().fit_transform(X)
        X = torch.FloatTensor(X)
        
        X_train, X_test = X[:num_train], X[num_train:]

        train_data = VectorDataset(X_train, S_train, Y_train)
        test_data = VectorDataset(X_test, S_test, Y_test)

        return train_data, test_data

def loadGerman(discrete=True):
    df = pd.read_csv('data/german/german.data', names=range(1,22), sep=' ')
    df = shuffle(df)
    num_train = int(0.8*df.shape[0])
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))
    
    category_cols = [1,3,4,6,7,8,9,10,11,12,14,15,16,17,18,19,20]
    continuous_cols = [2,5]
    sensitive_col = 13
    label_col = 21
    
    S = (df[sensitive_col].values>=30).astype(int)
    Y = (df[label_col]-1).values
    S = torch.LongTensor(S)
    Y = torch.LongTensor(Y)
    del df[sensitive_col]
    del df[label_col]
    
    S_train, S_test = S[:num_train], S[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    df[2] = pd.cut(df[2], bins=5, labels=False)
    df[5] = pd.cut(df[5], bins=5, labels=False)
    df = df.apply(lambda col: LabelEncoder().fit_transform(col))

    if discrete:
        X, T, D = getOneHot(df, df.columns)
        
        X = torch.FloatTensor(X)
        T = torch.LongTensor(T)
        D = list(D)
        
        X_train, X_test = X[:num_train], X[num_train:]
        T_train, T_test = T[:num_train], T[num_train:]
        
        train_data = VectorDataset(X_train, S_train, Y_train, T_train)
        test_data = VectorDataset(X_test, S_test, Y_test, T_test)
        train_data.D = D

        return train_data, test_data
    else:
        X = df.values
        X = MinMaxScaler().fit_transform(X)
        
        X = torch.FloatTensor(X)
        
        X_train, X_test = X[:num_train], X[num_train:]
        
        train_data = VectorDataset(X_train, S_train, Y_train)
        test_data = VectorDataset(X_test, S_test, Y_test)
        
        return train_data, test_data

def loadHealth(discrete=True):
    df = pd.read_csv('data/health.csv')
    df = df[df['YEAR_t'].isin(['Y2'])]
    df = df[(df['sexMISS'] == 0)&(df['age_MISS'] == 0)]
    df = shuffle(df).reset_index(drop=True)
    
    num_train = int(df.shape[0]*0.8)
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))
    
    age = df[['age_%d5' % (i) for i in range(0, 9)]].values.argmax(axis=1)
    sex = df[['sexMALE', 'sexFEMALE']].values.argmax(axis=1)
    
    Y = (df['DaysInHospital'] > 0).values.astype(int)
    drop_cols = ['age_%d5' % (i) for i in range(0, 9)] + ['sexMALE', 'sexFEMALE',  'sexMISS', 'age_MISS', 'trainset', 'DaysInHospital', 'MemberID_t', 'YEAR_t']
    
    df = df.drop(drop_cols, axis=1)
    df = getBinary(df, df.columns)

    df['sex'] = sex
    S = (age > 6).astype(int)
    
    S = torch.LongTensor(S)
    Y = torch.LongTensor(Y)
    S_train, S_test = S[:num_train], S[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]
    
    if discrete:
        X, T, D = getOneHot(df, df.columns)
        X = torch.FloatTensor(X)
        T = torch.LongTensor(T)
        D = list(D)
        
        X_train, X_test = X[:num_train], X[num_train:]
        T_train, T_test = T[:num_train], T[num_train:]
        
        train_data = VectorDataset(X_train, S_train, Y_train, T_train)
        test_data = VectorDataset(X_test, S_test, Y_test, T_test)
        train_data.D = D

        return train_data, test_data
    else:
        X = df.values
        X = MinMaxScaler().fit_transform(X)
        X = torch.FloatTensor(X)
        
        X_train, X_test = X[:num_train], X[num_train:]
        
        train_data = VectorDataset(X_train, S_train, Y_train)
        test_data = VectorDataset(X_test, S_test, Y_test)
        
        return train_data, test_data

def evaluate(Z_train, Z_test, S_train, S_test, Y_train, Y_test):
    cols = ['I_ZS', 'I_SY', 'I_ZY', 'sAUC', 'yAUC', 'DP']
    
    # 1. I_ZS
    I_ZS = computeMI(Z_test, S_test)

    # 2. I_SY
    S = np.concatenate([S_train, S_test]).reshape(-1,1)
    Y = np.concatenate([Y_train, Y_test])
    I_SY = mutual_info_classif(S, Y, discrete_features=[True], n_neighbors=5)[0]

    # 3. I_ZY
    I_ZY = computeMI(Z_test, Y_test)
    ans = [I_ZS, I_SY, I_ZY]

    clf = RandomForestClassifier()
    # 4. sAUC
    pred = clf.fit(Z_train, S_train).predict_proba(Z_test)[:,1]
    sAUC = roc_auc_score(S_test, pred)
    ans.append(sAUC)
    
    # 5. yAUC
    clf = clf.fit(Z_train, Y_train)
    pred = clf.predict_proba(Z_test)[:,1]
    yAUC = roc_auc_score(Y_test, pred)
    ans.append(yAUC)
    
    # 6. DP
    pred = clf.predict(Z_test)
    pred_0 = pred[S_test==0]
    pred_1 = pred[S_test==1]
    dp_gap = abs(pred_0.mean() - pred_1.mean())
    ans.append(dp_gap)

    return ans, cols

def computeMI(c, d, n_neighbors=5):
    n_samples = c.shape[0]
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()
    
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count
    
    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]
    
    kd = KDTree(c)
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.array(m_all) - 1.0
    
    mi = (digamma(n_samples)
        + np.mean(digamma(k_all))
        - np.mean(digamma(label_counts))
        - np.mean(digamma(m_all + 1))
        )
    
    return max(0, mi)
