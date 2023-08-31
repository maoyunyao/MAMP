import os
import argparse
import numpy as np
from tqdm import tqdm

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300



def read_data(data_path, name, max_body=4, num_joint=25):  # top 2 body
    filename, action_idx = name.split('_')
    action_idx = int(action_idx)
    seq_data = np.loadtxt('{}/skeleton/{}'.format(data_path, filename))
    label = np.loadtxt('{}/label/{}'.format(data_path, filename), delimiter=',')
    start, end = int(label[action_idx][1]), int(label[action_idx][2])
    
    data = seq_data[start: end, :]  # num_frames * 150
    # data = data.reshape(data.shape[0], 2, 25, 3)  # num_frame, num_body, num_joint, xyz
    # data = data.transpose(3, 0, 2, 1)  # xyz, num_frame, num_joint, num_body
    return data

def gendata(data_path, benchmark='XView', part='test'):
    # Read cross_subject_v1.txt and cross_view_v1.txt to obtain training_views training_subjects
    with open('{}/cross-view.txt'.format(data_path), 'r') as f:
        lines = f.readlines()
        training_views = lines[1].strip('\n').split(', ')
    with open('{}/cross-subject.txt'.format(data_path), 'r') as f:
        lines = f.readlines()
        training_subjects = lines[1].strip('\n').split(', ')

    sample_names = []
    sample_joints = []
    sample_labels = []
    for filename in os.listdir('{}/skeleton'.format(data_path)):
        if benchmark == 'XView':
            istraining = (filename[:-4] in training_views)
        elif benchmark == 'XSub':
            istraining = (filename[:-4] in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'test':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            video_labels = np.loadtxt('{}/label/{}'.format(data_path, filename), delimiter=',')
            for idx in range(video_labels.shape[0]):
                inst_name = '{}_{}'.format(filename, str(idx))
                inst_label = int(video_labels[idx][0] - 1)
                inst_data = read_data(data_path, inst_name, max_body=max_body_kinect, num_joint=num_joint)

                if inst_data.sum() != 0:
                    sample_names.append(inst_name)
                    sample_labels.append(inst_label)
                    sample_joints.append(inst_data)
                else:
                    print('Find all zero instance.')


    fp = np.zeros((len(sample_joints), max_frame, max_body_true * num_joint * 3), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_names)):
        data = sample_joints[i]
        fp[i, 0:min(data.shape[0], max_frame), :] = data[0:min(data.shape[0], max_frame), :]  # num_frame 太大会截断！

    return fp, sample_labels


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 51))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PKU-MMD-v1 Data Converter.')
    parser.add_argument('--data_path', default='../pku_raw/v1')
    arg = parser.parse_args()
    
    benchmark = ['XSub','XView']
    for b in benchmark:
        x_train, y_train = gendata(arg.data_path, benchmark=b, part='train')
        x_test, y_test = gendata(arg.data_path, benchmark=b, part='test')

        y_train = one_hot_vector(y_train)
        y_test = one_hot_vector(y_test)

        save_name = 'PKUv1_%s.npz' % b
        np.savez(save_name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        
        
