import os

import pickle

import pandas as pd

import numpy as np
from scipy import ndimage

from sklearn.decomposition import PCA


config = dict()
config['data_dir'] = os.path.abspath('D:\\brain\\dataset\\Canada\\PCA')
config['data_filenames'] = os.listdir(config['data_dir'])


def _read_file(filename):
    df = pd.read_csv(filename, sep=',', header=0, usecols=[1, 2, 3, 14, 15, 16, 17])
    df = df[df.corpuscallosum == 1]
    return df


def _resample(dataset):
    numpy_data = [data.to_numpy() for data in dataset]
    lengths = [len(data) for data in dataset]

    resampled_data_list = list()

    for index, data in enumerate(numpy_data):
        scale_factor = (min(lengths)) / lengths[index]
        tmp = ndimage.interpolation.zoom(data, [scale_factor, 1], order=2)
        resampled_data_list.append(tmp)

    return resampled_data_list


def _perform_pca(X, subject_num=None, **kwargs):
    pca = PCA()
    X = X.T
    pca.fit(X)

    pca_mean = pca.mean_
    pca_mean = pca_mean.reshape((1, len(pca_mean)))
    pca_components = pca.components_.T
    pca_variance = pca.explained_variance_
    pca_explained_variance = pca.explained_variance_ratio_

    count = len(pca_variance)

    mode_scores = list()
    for i in range(len(X)):
        subject = X[i] - pca_mean
        score = np.dot(subject, pca_components)
        mode_scores.append(score)

    if subject_num is not None:
        sd = np.std(mode_scores, axis=0)
        mean = np.average(mode_scores, axis=0)
        s = X[subject_num] - pca_mean
        score = np.dot(s, pca_components)
        score = score[0][0:count]
        score_normalized = _convert_score(score, sd, mean)

        if kwargs:
            for arg in kwargs.values():
                number_of_pca = arg
            recon = _reconstruct_subjects_from_pca(score_normalized, pca_components, pca_mean, number_of_pca)
            return recon.flatten().reshape((recon.shape[1]//6, 6))
        return score_normalized
    return


def _convert_score(scores, sd, mean):
    score_1 = list()
    mean = mean.flatten()
    sd = sd.flatten()
    for i in range(len(scores)):
        score_1.append((scores[i] - mean[i]) / sd[i])
    return score_1


def _reconstruct_subjects_from_pca(scores, components, mean, num):
    if num > len(scores):
        raise ValueError("Number of input PCs are more than the total number of components!")
    return np.dot(scores[0:num+1], components.T[0:num+1]) + mean


def _save_exdata(data_array, pca_data_array, filename):
    with open(filename+'.exdata', 'w') as exdata:
        exdata.writelines(" Group name: DTI\n")
        exdata.writelines(" #Fields= 2\n")
        exdata.writelines(" coordinates, coordinate, rectangular cartesian, #Components=3\n")
        exdata.writelines("   x.  Value index= 1, #Derivatives=0\n")
        exdata.writelines("   y.  Value index= 2, #Derivatives=0\n")
        exdata.writelines("   z.  Value index= 3, #Derivatives=0\n")
        exdata.writelines(" vector, field, rectangular cartesian, #Components=3\n")
        exdata.writelines("   1.  Value index= 4, #Derivatives=0\n")
        exdata.writelines("   2.  Value index= 5, #Derivatives=0\n")
        exdata.writelines("   3.  Value index= 6, #Derivatives=0\n")

    for index, value in enumerate(data_array):
        node = index + 1
        with open(filename+'.exdata', 'a') as exdata:
            exdata.writelines(" Node:     %s\n" % node)
            exdata.writelines("   %s %s %s\n" % (data_array[index][0], data_array[index][1], data_array[index][2]))
            exdata.writelines("   %s %s %s\n" % (pca_data_array[index][3], pca_data_array[index][4], pca_data_array[index][5]))


def _main(subjectnum=None):
    # data_list = list()
    # for filename in config['data_filenames']:
    #     data = _read_file(os.path.join(config['data_dir'], filename))
    #     data_list.append(data)

    with open('D:\\brain\\dataset\\Canada\\PCA\\data.pkl', 'rb') as f:
        data_list = pickle.load(f)

    resampled_data = _resample(data_list)
    resampled_data = [resampled_data[i][:, :-1] for i in range(len(resampled_data))]
    X = np.zeros((resampled_data[0].shape[0]*resampled_data[0].shape[1], len(resampled_data)))

    for i in range(len(X[1])):
        X[:, i] = resampled_data[i].flatten()

    return resampled_data, _perform_pca(X, subject_num=subjectnum, number_of_pcs_for_reconstruction=1)


if __name__ == '__main__':
    for i in range(10):
        print('Performing PCA for subject %s' % (i + 1))
        rd, reconstructed_subject = _main(subjectnum=i)
        exfilename = 'Subject_%s_PC_1' % (i + 1)
        exfilepath = 'D:\\brain\\dataset\\Canada\\PCA\\' + exfilename
        print('Saving Exdata...')
        _save_exdata(rd[i], reconstructed_subject, exfilepath)
    print('done')

