from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
#from sklearn import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors as mcolors

import seaborn as sns
import glob
import os

import scipy
from scipy.stats import pearsonr

import importlib
from datapaths import *


fpsMovie = [['After_The_Rain',23.976],
            ['Attitude_Matters',29.97],
            ['Barely_legal_stories',23.976],
            ['Between_Viewings',25],
            ['Big_Buck_Bunny',24],
            ['Chatter',24],
            ['Cloudland',25],
            ['Damaged_Kung_Fu',25],
            ['Decay',23.976],
            ['Elephant_s_Dream',24],
            ['First_Bite',25],
            ['Full_Service',29.97],
            ['Islands',23.976],
            ['Lesson_Learned',29.97],
            ['Norm',25],
            ['Nuclear_Family',23.976],
            ['On_time',30],
            ['Origami',24],
            ['Parafundit',24],
            ['Payload',25],
            ['Riding_The_Rails',23.976],
            ['Sintel',24],
            ['Spaceman',23.976],
            ['Superhero',29.97],
            ['Tears_of_Steel',24],
            ['The_room_of_franz_kafka',29.786],
            ['The_secret_number',23.976],
            ['To_Claire_From_Sonny',23.976],
            ['Wanted',25],
            ['You_Again',29.97]]

contmoviesfps = pd.DataFrame(fpsMovie,columns=['name','fps'])
contmoviesfps.set_index('name', inplace=True)
contmoviesfps.index.name = None
#contmoviesfps.index
#contmoviesfps.values
contmoviesfps['f'] = np.round(contmoviesfps['fps'])

def getfps(movname):
    return contmoviesfps.loc[movname ]['f']

def getfps(movname):
    return contmoviesfps.loc[movname ]['f']

def getDevMovieNames():
    return [i[0] for i in fpsMovie]

movgroups_wodecay = {
    0:['You_Again','Damaged_Kung_Fu','The_secret_number','Spaceman'],
    1:['Cloudland','Origami','Riding_The_Rails','Tears_of_Steel','Sintel'],
    2:['On_time','Elephant_s_Dream','Norm','Big_Buck_Bunny','Chatter','Full_Service'],
    3:['Islands','To_Claire_From_Sonny','Nuclear_Family','After_The_Rain','Parafundit'],
    4:['The_room_of_franz_kafka','Attitude_Matters','Lesson_Learned','Superhero'],
    5:['First_Bite','Wanted','Between_Viewings','Barely_legal_stories','Payload']
}

movgroups = {
    0:['You_Again','Damaged_Kung_Fu','The_secret_number','Spaceman'],
    1:['Cloudland','Origami','Riding_The_Rails','Tears_of_Steel','Sintel'],
    2:['On_time','Elephant_s_Dream','Norm','Big_Buck_Bunny','Chatter','Full_Service'],
    3:['Islands','To_Claire_From_Sonny','Nuclear_Family','After_The_Rain','Parafundit'],
    4:['The_room_of_franz_kafka','Attitude_Matters','Lesson_Learned','Superhero'],
    5:['First_Bite','Wanted','Between_Viewings','Barely_legal_stories','Payload'],
    6:['Decay']
}

mov2groups = {
    0:['Decay'],
    1:['You_Again','Damaged_Kung_Fu','The_secret_number','Spaceman'],
    2:['Cloudland','Origami','Riding_The_Rails','Tears_of_Steel','Sintel'],
    3:['On_time','Elephant_s_Dream','Norm','Big_Buck_Bunny','Chatter','Full_Service'],
    4:['Islands','To_Claire_From_Sonny','Nuclear_Family','After_The_Rain','Parafundit'],
    5:['The_room_of_franz_kafka','Attitude_Matters','Lesson_Learned','Superhero'],
    6:['First_Bite','Wanted','Between_Viewings','Barely_legal_stories','Payload'],
}


def gettraintestmovielist(mlist,groups=movgroups):
    testlist = groups[mlist]
    trainlist = []
    for idx, group in enumerate(groups):
        if idx != mlist:
            for g in groups[idx]:
                trainlist.append(g)
    return trainlist, testlist

################# ANNOTATIONS #################################
def getFullArousalDf(movname):
    filename = os.path.join(med2016annotationsFolder, movname+"_Arousal.txt")
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    df.set_index('Time',inplace=True)
    df.rename(columns={'Mean':'MeanArousal','Std':'StdArousal'},inplace=True)
    return df

def getFullValenceDf(movname):
    filename = os.path.join(med2016annotationsFolder, movname+"_Valence.txt")
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    df.set_index('Time',inplace=True)
    df.rename(columns={'Mean':'MeanValence','Std':'StdValence'},inplace=True)
    return df

def getAnnotationDf(movname,folder=med2017annotationsFolder):
    filename = os.path.join(folder, movname + '-MEDIAEVAL2017-valence_arousal.txt')
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    return df

def getFearDf(movname):
    filename = os.path.join(med2017fearFolder, movname + '-MEDIAEVAL2017-fear.txt')
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    return df

def avfear(mov = 'Sintel'):
    plt.figure(figsize=(10,8))
    plt.axis([-1, 1500, -1, 1])
    avdf = getAnnotationDf(mov)
    avdf[['MeanValence','MeanArousal']].plot(ax=plt.gca(),style='-',title=mov)
    feardf = getFearDf(mov)
    feardf[['Fear']].plot(ax=plt.gca(),title=mov,style='-')
    return avdf,feardf


################## FEATURES ################################
def getfacedf(moviename):
    filename = facesfolder + moviename +'.mp4-faces.txt'
    df = pd.read_csv(filename,sep=' ')

    df['topratio'] = df['top']/df['hframe']
    df['bottomratio'] = df['bottom']/df['hframe']
    df['leftratio'] = df['left']/df['wframe']
    df['rightratio'] = df['right']/df['wframe']
    df['fheight'] =  (df['bottom']-df['top'])/df['hframe']
    df['fwidth'] =  (df['right']-df['left'])/df['wframe']
    df['farea'] = df['fheight']*df['fwidth']
    df['fcx'] = df['fheight'] / 2
    df['fcy'] = df['fwidth'] / 2

    #df[ df['noface']>0]
    cols = ['noface','fcx','fcy','topratio','bottomratio','leftratio','rightratio','fheight','fwidth','farea']
    return df[cols]

'''
def getfacelandmarksdf(moviename):
    filename = faceslandmarksfolder + moviename + '.mp4-faces-landmarks.txt'

    df = pd.read_csv(filename, sep=' ')
    ## This columns are in wrong order in the file e.g noface is before hframe
    cols = ["id", "hframe", "wframe", "noface", "top", "left", "bottom", "right"]

    df = df.drop(cols, axis=1)

    df = df.fillna(0)

    # df = df[df['0']>0]

    return df
'''

def getLowFeatureDf(movname):
    fname = movname +'.mp4continous_features.txt'
    df = pd.DataFrame(np.genfromtxt( os.path.join(pathcontfeatures,fname)))
    df.columns = ['time','framemean','huemean','satmean','valmean', 'redmean','greenmean','bluemean', 'lummean','motion']
    return df

def getLowFeature10SecDf(movname):
    pdf = getLowFeatureDf(movname)

    dfwindow = pdf.rolling(10).mean()[9::5]
    dfwindow.reset_index(inplace=True)
    dfwindow.drop('index',axis=1,inplace=True)
    dfwindow.drop('time',axis=1,inplace=True)
    return dfwindow

def getMovieListLowFeatFearDf(movieNames):
    X = getLowFeature10SecDf(movieNames[0])
    y = getFearDf(movieNames[0]).Fear[:len(X)]

    for mov in movieNames[1:]:
        tX=getLowFeatureDf(mov)
        ty=getFearDf(mov).Fear[:len(tX)]
        X = X.append(tX)
        y = y.append(ty)
        if (X.shape != y.shape):
            print( mov, X.shape, y.shape)
    return X,y

def df2mat(df):
    return df.as_matrix().reshape((len(df),))

def getColorCenters(movie):
    df = pd.read_csv(colfold+movie+'-color-info.txt',  sep='\t', index_col=0 )
    df = df.infer_objects()
    return df

######################## VISUALIZATIONS ################################

def displayframe(movie,number):
    filename = framesfolder+movie +'.mp4-'+ '{0:05d}'.format(number) +'.jpg'

    img=mpimg.imread(filename)
    plt.figure(figsize=(20,30))
    plt.grid(False)

    imgplot = plt.imshow(img)
    plt.show()

def displayAVscores(movieNames):
    fix, axes = plt.subplots(figsize=(20,16))
    for ii, mov in enumerate(movieNames):
        plt.subplot(6,5,ii+1)
        #plt.axis([-1, 1000, -1, 1])
        plt.ylim([-1,1])
        dfa = getAnnotationDf(mov,med2017annotationsFolder)
        dfa[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)

def displayFear(movieNames):
    fix, axes = plt.subplots(figsize=(20,16))
    for ii, mov in enumerate(movieNames):
        plt.subplot(6,5,ii+1)
        df = getFearDf(mov)
        df[['Fear']].plot(ax=plt.gca(),title=mov)

def displayAVFear(movieNames):
    fix, axes = plt.subplots(figsize=(20, 16))
    for ii, mov in enumerate(movieNames):
        plt.subplot(6, 5, ii + 1)
        plt.ylim([-1, 1])
        df = getAnnotationDf(mov)
        feardf = getFearDf(mov)

        df[['MeanValence', 'MeanArousal']].plot(ax=plt.gca(), title=mov)
        feardf[['Fear']].plot(ax=plt.gca(), title=mov)
        farr = feardf[feardf['Fear'] > 0]['Id'].as_matrix()
        for i in farr:
            plt.axvline(x=i, linestyle='-', color='red')


def plotmovieAV(mov):
    dfa = getAnnotationDf(mov)
    f, axs = plt.subplots(2, 1, figsize=(15, 15))

    plt.subplot(2, 1, 1)
    dfa[['MeanValence']].plot(ax=plt.gca(), title=mov)

    plt.subplot(2, 1, 2)
    dfa[['MeanArousal']].plot(ax=plt.gca(), title=mov)


def displayColorPallette(framescolor):
    xi=0
    ymax=1
    xstep = 1
    ystep= 0.20 #1.0/ k
    linewidth=20

    f , axs = plt.subplots(3,1,figsize=(10,10))

    plt.subplot(3,1,1)
    plt.xlabel('Time/Segment')
    plt.ylabel('HSV sorted')

    for frame in framescolor:
        ymax=1
        sfrm = sorted([tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])) for color in frame])
        for c in sfrm:
            plt.axvline(x=xi, ymax=ymax, ymin=ymax-ystep,linewidth=25, color=c)
            ymax -= ystep
        xi=xi+xstep

    plt.subplot(3,1,2)
    xi=0
    plt.xlabel('Time/Segment')
    plt.ylabel('RGB sorted')
    for frame in framescolor:
        ymax=1
        frm = sorted([tuple(mcolors.to_rgba(color)[:3]) for color in frame])
        for c in frm:
            plt.axvline(x=xi, ymax=ymax, ymin=ymax-ystep,linewidth=25, color=c)
            ymax -= ystep
        xi=xi+xstep

    plt.subplot(3,1,3)
    xi=0
    plt.xlabel('Time/Segment')
    plt.ylabel('RGB colors (5)')
    for frame in framescolor:
        ymax=1
        for c in frame:
            plt.axvline(x=xi, ymax=ymax, ymin=ymax-ystep,linewidth=25, color=c)
            ymax -= ystep
        xi=xi+xstep


def displaymoviecolor(movie):
    df = getColorCenters(movie)
    framescolor = df.values
    displayColorPallette(framescolor)


####################### MODELS and METRICS ###############################
def holt_winters_second_order_ewma( x, span, beta ):
    N = x.size
    alpha = 2.0 / ( 1 + span )
    s = np.zeros(( N, ))
    b = np.zeros(( N, ))
    s[0] = x[0]
    for i in range( 1, N ):
        s[i] = alpha * x[i] + ( 1 - alpha )*( s[i-1] + b[i-1] )
        b[i] = beta * ( s[i] - s[i-1] ) + ( 1 - beta ) * b[i-1]
    return s

def getMetrics(y,y_pred):
    # calculate MAE using scikit-learn
    #mae = metrics.mean_absolute_error(ytestarray, y_pred)
    #print("MAE score: {:.5f}".format(mae))

    mse = metrics.mean_squared_error(y, y_pred)
    # calculate MSE using scikit-learn
    print("MSE score: {:.5f}".format(mse))

    # calculate RMSE using scikit-learn
    #print("RMSE: {:.5f}".format(np.sqrt(metrics.mean_squared_error(ytestarray, y_pred))))

    print("Pearson score:")
    prs = pearsonr(y,y_pred)
    print(prs)

    return mse,prs

def evaluate_pipe(pipe,trainX,trainy,testX,testy):

    ytrainarray = trainy.as_matrix().reshape((len(trainy),))
    ytestarray = testy.as_matrix().reshape((len(testy),))

    pipe.fit(trainX, ytrainarray)

    print("Train score: {:.2f}".format(pipe.score(trainX, ytrainarray)))
    print("Test score: {:.2f}".format(pipe.score(testX, ytestarray)))

    y_pred = pipe.predict(testX)

    mse, prs = getMetrics(ytestarray,y_pred)

    return y_pred,mse,prs[0],pipe

def define_pipelines():
    pipe_visual_valence = make_pipeline(
        StandardScaler(),
        SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
            gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True,
            tol=0.001, verbose=False))

    pipe_visual_arousal = make_pipeline(
        StandardScaler(),
        SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
            gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True,
            tol=0.001, verbose=False))

    # Audio
    pipe_audio_valence = make_pipeline(
        StandardScaler(copy=True, with_mean=True, with_std=True),
        SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
            kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))

    pipe_audio_arousal = make_pipeline(
        StandardScaler(copy=True, with_mean=True, with_std=True),
        #PCA(n_components=800),
        SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
            kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))

    # FC16 -->deep fetures
    pipe_deep_valence = make_pipeline(
        StandardScaler(),
        SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
            gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True,
            tol=0.001, verbose=False))

    pipe_deep_arousal = make_pipeline(
        StandardScaler(),
        SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
            gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True,
            tol=0.001, verbose=False))

    # Low Level Features
    pipe_llf_valence = make_pipeline(
        StandardScaler(),
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features=9, max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False))

    pipe_llf_arousal = make_pipeline(
        StandardScaler(),
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features=9, max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False))

    # Face Features
    pipe_face_valence = make_pipeline(
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=100,
               max_features=10, max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False))

    pipe_face_arousal  = make_pipeline(
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=100,
               max_features=10, max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False))
    return 0

################### CONVERSIONS ###################
#in minutes
def frame2minutes(moviename,frameno=1,isframepersec=True):
    fps = getfps(moviename)
    if isframepersec:
        secs = frameno
    else:
        secs = frameno/fps
    mins = secs / 60
    sec = (secs%60)
    return mins, sec

#in seconds
def frame2time(moviename,frameno=1,isframepersec=True):
    if (isframepersec):
        fps = getfps(moviename)
        return frameno/fps
    return frameno


def frame2sampleid(moviename,frameno=1,isframepersec=True):
    return frame2time(moviename,frameno)/5


def time2frame(moviename,hours=0,minutes=0,secs=0,isframepersec=True):
    totalsec= hours*60*60 + minutes*60 + secs
    if isframepersec:
        return totalsec

    fps = getfps(moviename)
    return totalsec*fps

def sec2frame(moviename,totalsec=1,isframepersec=True):
    if isframepersec:
        return totalsec
    fps = getfps(moviename)
    return totalsec*fps

def sample2frame(moviename,sampleid):
    totalsec=sampleid*5
    return sec2frame(moviename,totalsec)
