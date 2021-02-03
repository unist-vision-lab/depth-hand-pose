from model.model_3d import CPMAllPose
import torch
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2

import torch.nn as nn
import torch.optim as optim
import math

import matplotlib.pyplot as plt
import pickle
import copy
from datetime import datetime

device = 'cuda:0'
num_joints = 21



def getDatasetPath(dataset_name, root):
    if dataset_name == "obman":
        train_annotation_file = root + "train/result_training.pickle"
        test_annotation_file = root + "test/result_testing.pickle"

        # 140,240 frames
        with open(train_annotation_file, 'rb') as f:
            data = pickle.load(f)
            train_images_paths = data['images']
            train_annotations = data['annotations']

        filelist = ["00029634.jpg", "00045305.jpg", "00047433.jpg", "00119666.jpg", "00186570.jpg"]

        temp = []
        for idx, i in enumerate(train_images_paths):
            for j in filelist:
                if i['file_name'] == j:
                    temp.append(idx)

        ## annotations
        for idx, i in enumerate(temp):
            # new = int(i)+idx
            train_images_paths.pop(int(i) - idx)
            train_annotations.pop(int(i) - idx)


        # 6285 frames
        with open(test_annotation_file, 'rb') as f:
            data = pickle.load(f)
            test_images_paths = data['images']
            test_annotations = data['annotations']


        return train_images_paths, train_annotations, test_images_paths, test_annotations



class Dataset(Dataset):
    def __init__(self, method, dataset_name):
        self.method = method
        self.dataset_name = dataset_name  # dataset directory 이름과 일치하도록 지정
        self.root = '/data/pose_estimation/'
        self.image_save_root = '/result_image/obman_100epoch_32batch/'
        train_images_paths, train_annotations, test_images_paths, test_annotations = getDatasetPath(dataset_name=self.dataset_name, root=self.root + self.dataset_name +'/')


        if method == 'train':
            self.dataset_len = len(train_images_paths)
            self.images_paths = train_images_paths
            self.annotations = train_annotations


        elif method == 'test':
            self.dataset_len = len(test_images_paths)
            self.images_paths = test_images_paths
            self.annotations = test_annotations

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.images_paths)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        file_png = self.images_paths[idx]['file_name']
        self.images_paths[idx]['file_name'] = file_png.replace("jpg", "png")
        img_path = self.root + self.dataset_name + "/" + str(self.method)+ "/"+ 'depth/' + self.images_paths[idx]['file_name']
        ori_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = copy.deepcopy(ori_img)
    #    b, g, r = cv2.split(img)
    #    img = cv2.merge([r, g, b])


        bbox = np.array(copy.deepcopy(self.annotations[idx]['bbox']))
        xyz = copy.deepcopy(self.annotations[idx]['xyz'])
        intrinsic = copy.deepcopy(self.images_paths[idx]['param'])
        depth_root = copy.deepcopy(self.annotations[idx]['uvd'][:, 2:][9])
        ori_uvd = copy.deepcopy(self.annotations[idx]['uvd'])
        uvd = copy.deepcopy(self.depth_normalization(ori_uvd, depth_root))

        x, y, w, h = bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]
        cropped_img = img[y: h, x: w]
        temp_img = cv2.resize(cropped_img, dsize=(256, 256))
        img = transform1(temp_img)

        uvd[:, 0] = uvd[:, 0] - x
        uvd[:, 1] = uvd[:, 1] - y

        uvd[:, 1] = (256 / cropped_img.shape[0]) * uvd[:, 1]
        uvd[:, 0] = (256 / cropped_img.shape[1]) * uvd[:, 0]

        # fig, ax = plt.subplots(1)
        # ci = np.array(img.permute(1, 2, 0))
        # ax.imshow(ci)
        # plot_hand(uvd, ax)
        # plt.show()

        return img, xyz, uvd, img_path, intrinsic, depth_root, bbox

    def depth_normalization(self, new_uvds, root_depth):
        new_uvds[:, 2:] = new_uvds[:, 2:] * (5e-1 / root_depth)
        return new_uvds

class Trainer(object):
    #epochs, batchSize, learningRate, dataset_name, 'test'
    def __init__(self, epochs, batch_size, lr, dataset_name, method, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr

        self.dataset_name = dataset_name
        self.method = method
        self._build_model()

        self.cost = nn.MSELoss()
        self.optimizer = optim.Adam(self.poseNet.parameters(), lr=self.learning_rate)

        dataset = Dataset(method=method, dataset_name=dataset_name)
        self.datalen = dataset.__len__()
        self.root = dataset.root
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        # Load of pretrained_weight file
        weight_PATH = './result_model_depth/2021_1_4_0_model.pth'
        self.poseNet.load_state_dict(torch.load(weight_PATH), strict=False)
        self.date = "_".join([str(datetime.today().year), str(datetime.today().month), str(datetime.today().day)])
        # self.date = '201205'

        print("Training...")

    def _build_model(self):
        # 2d pose estimator
        poseNet = CPMAllPose()
        self.poseNet = poseNet.to(device)
        self.poseNet.train()

        print('Finish build model.')

    def skeleton2heatmap(self, _heatmap, ori_keypoint_targets, img, img_path):

        heatmap_gt = torch.zeros(_heatmap.shape, device=_heatmap.device)
        keypoint_targets = (((ori_keypoint_targets)) // 8)
        for i in range(keypoint_targets.shape[0]):
            for j in range(keypoint_targets.shape[1]):
                x = int(keypoint_targets[i, j, 1]) #가로
                y = int(keypoint_targets[i, j, 0]) #세로

                if x >= heatmap_gt.shape[2]:
                    print("=> x index error : ", x, "/ batch : ", i, " img_path : ", img_path[i])
                    # os.system("mv "+img_path[i]+" /disk/obman/train/error_rgb/" )

                elif y >= heatmap_gt.shape[3]:
                    print("=> x index error : ", x, "/ batch : ", i, " img_path : ", img_path[i])
                    # os.system("mv "+img_path[i]+" /disk/obman/train/error_rgb/" )

                elif x >= heatmap_gt.shape[2] and y >= heatmap_gt.shape[3]:
                    print("=> x index error : ", x, "/ batch : ", i, " img_path : ", img_path[i])
                    # os.system("mv "+img_path[i]+" /disk/obman/train/error_rgb/" )

                else:
                    heatmap_gt[i, j, x, y] = 1

        heatmap_gt = heatmap_gt.detach().cpu().numpy()

        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                heatmap_gt[i, j, :, :] = cv2.GaussianBlur(heatmap_gt[i, j, :, :], ksize=(3, 3), sigmaX=2, sigmaY=2) * 9 / 1.1772
        heatmap_gt = torch.FloatTensor(heatmap_gt).to(device)

        return heatmap_gt

    def train(self):
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch % 10 == 0:
                torch.save(self.poseNet.state_dict(), './result_model_depth/' + self.dataset_name + '/estimation/' + "_".join([str(self.date), str(epoch),'model.pth']))

            for batch_idx, samples in enumerate(self.dataloader):
                img, xyz, uvd, img_path, intrinsic, depth_root, bbox = samples #torch.Size([16, 480, 640, 3])
                heatmapsPoseNet, pose_3ds = self.poseNet(img.to(device).float())
                gt_heatmap = self.skeleton2heatmap(heatmapsPoseNet, uvd, img, img_path)

                heatmap_cost_result = self.cost(heatmapsPoseNet, gt_heatmap)
                gt_cost_result = self.cost(pose_3ds, uvd[:, :, 2:].squeeze().to(device).float())
                all_cost = heatmap_cost_result + gt_cost_result
                self.optimizer.zero_grad()
                all_cost.backward()
                self.optimizer.step()

                ##Write train result

                if (batch_idx + 1) % 20 == 0:
                    with open('./result_model_depth/' + self.dataset_name + '/training_log/'+"_".join([str(self.date),'log.txt']), 'a') as f:
                        f.write('Epoch {:4d}/{} Batch {}/{} estimation_All_Cost: {:.6f} / heat_cost: {:.6f} gt_cost: {:.7f}  \n'.format(
                            epoch, self.epochs, (batch_idx + 1), len(self.dataloader),
                            all_cost.item(), heatmap_cost_result.item(), gt_cost_result.item()
                        ))
                    print('Epoch {:4d}/{} Batch {}/{} estimation_Cost: {:.6f} / heat_cost: {:.6f} gt_cost: {:.7f} '.format(
                        epoch, self.epochs, (batch_idx + 1), len(self.dataloader),
                        all_cost.item(), heatmap_cost_result.item(), gt_cost_result.item()
                    ))

        print('Finish training.')


class Tester(object):
    def __init__(self, batch_size, dataset_name, method, num_workers):
        self._build_model()
        self.dataset_name = dataset_name
        self.pretrained_model = '/data/pose_estimation'
        self.mse_all_img = 0
        self.per_image = 0

        dataset = Dataset(method=method, dataset_name=dataset_name)
        self.save_img_root = dataset.image_save_root
        self.datalen = dataset.__len__()
        self.root = dataset.root
        self.error_check_x = 0
        self.error_check_y = 0
        self.error_check_z = 0


        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size,  shuffle=False)

        # Load of pretrained_weight file #100epoch32bat/
        weight_PATH = self.pretrained_model + '/result_model_depth/obman/estimation/' + '2021_1_6_70_model.pth' #92
        self.poseNet.load_state_dict(torch.load(weight_PATH), strict=False)

        print("Testing...")

    def _build_model(self):
        # 2d pose estimator
        poseNet = CPMAllPose()
        self.poseNet = poseNet.to(device)

        print('Finish build model.')


    def heatmap2skeleton(self, heatmapsPoseNet):
        skeletons = np.zeros((heatmapsPoseNet.shape[0], heatmapsPoseNet.shape[1], 2))
        for m in range(heatmapsPoseNet.shape[0]):
            for i in range(heatmapsPoseNet.shape[1]):
                u, v = np.unravel_index(np.argmax(heatmapsPoseNet[m][i].cpu().detach().numpy()), (32, 32))
                skeletons[m, i, 0] = v * 8
                skeletons[m, i, 1] = u * 8

        return skeletons

    def uvd_to_xyz2(self, uvd):
        cam = [480., 480., 128., 128.]
        z = uvd[:, 2]
        x = (uvd[:, 0] - cam[2]) * z / cam[0]
        y = (uvd[:, 1] - cam[3]) * z / cam[1]
        return np.array(torch.stack([x, y, z], dim=1))

    def test(self):
        for batch_idx, samples in tqdm.tqdm(enumerate(self.dataloader)):
            img, xyz, uvd, img_path, intrinsic, depth_root, bbox = samples
            # heatmaps, pose_3ds
            heatmapsPoseNet, pose_3ds = self.poseNet(img.to(device))
            skeletons_in = self.heatmap2skeleton(heatmapsPoseNet)

            new_3d_pose = pose_3ds.cpu().detach().numpy().reshape(skeletons_in.shape[0], skeletons_in.shape[1], 1)

            ### Visualization
            for a, img_n in enumerate(img):
                fig, ax = plt.subplots(1)
                ci = np.array(img_n.permute(1, 2, 0))
                ax.imshow(ci)
                uvd_n = np.concatenate((skeletons_in[a], new_3d_pose[a]), axis=1)
                plot_hand(uvd_n, ax)
               # fig.savefig('00001.png', dpi=fig.dpi)
                plt.show()

            for per_batch in range(len(skeletons_in)):

                estimated_uvd = np.c_[skeletons_in[per_batch], new_3d_pose[per_batch]]
                ori_estimated_uvd = copy.deepcopy(estimated_uvd)
                x, y, w, h = bbox[per_batch][0], bbox[per_batch][1], bbox[per_batch][2], bbox[per_batch][3]
                ### Error calculation

                estimated_uvd = torch.from_numpy(estimated_uvd)
                estimated_uvd[:, 1] = estimated_uvd[:, 1] / (256.0 / h.float())
                estimated_uvd[:, 0] = estimated_uvd[:, 0] / (256.0 / w.float())
                estimated_uvd[:, 0] = estimated_uvd[:, 0] + x
                estimated_uvd[:, 1] = estimated_uvd[:, 1] + y

                estimated_uvd[:, 2] = estimated_uvd[:, 2] * depth_root[per_batch] / 5e-1
                estimated_xyz = self.uvd_to_xyz2(estimated_uvd)
                estimated_xyz = torch.from_numpy(estimated_xyz).float()
                estimated_xyz[:, 0] = estimated_xyz[:, 0] * -1


                max_error_idx = 0
                fig_x = np.arange(21)
                joints_error_idx = list()
                joints_number_list = ['Wrist(0)', 'Metacarpals(1)', 'ThumbProximal(2)', 'ThumbDistal(3)', 'ThumbTip(4)', 'IndexKnuckle(5)', 'IndexMiddle(6)', 'IndexDistal(7)', 'IndexTip(8)',
                                      'MiddleKnuckle(9)', 'MiddleMiddle(10)', 'MiddleDistal(11)', 'MiddleTip(12)', 'RingKnuckle(13)', 'RingMiddle(14)', 'RingDistal(15)', 'RingTip(16)',
                                      'PinkyKnuckle(17)', 'PinkyMiddle(18)', 'PinkyDistal(19)', 'PinkyTip(20)']

                for per_joints in range(estimated_xyz.shape[0]):
                    x = (xyz[per_batch][per_joints][0] - estimated_xyz[per_joints][0]) ** 2
                    y = (xyz[per_batch][per_joints][1] - estimated_xyz[per_joints][1]) ** 2
                    z = (xyz[per_batch][per_joints][2] - estimated_xyz[per_joints][2]) ** 2

                    xy_sum = x + y + z
                    x_sum = math.sqrt(x)
                    y_sum = math.sqrt(y)
                    z_sum = math.sqrt(z)

                    xy_sum = math.sqrt(xy_sum)
                    # print(" ERROR VALUE : (idx) ", per_joints, "(value) ", xy_sum)
                    if per_joints == 0:
                        max_error_idx = dict(idx=0, value=xy_sum)
                    elif xy_sum > max_error_idx['value']:
                        max_error_idx = dict(idx=per_joints, value=xy_sum)

                    joints_error_idx.append(xy_sum)

                    self.error_check_x += x_sum / 21
                    self.error_check_y += y_sum / 21
                    self.error_check_z += z_sum / 21
                    self.mse_all_img += xy_sum / 21


        final_result = self.mse_all_img / self.datalen
        x_error = self.error_check_x / self.datalen
        y_error = self.error_check_y / self.datalen
        z_error = self.error_check_z / self.datalen
        print("Number of dataset :", self.datalen)  # * 1000
        print("Trained model Result :", final_result * 1000) #* 1000
        print("X error : ", x_error * 1000, " Y error : ", y_error* 1000, " Z error : ", z_error* 1000)
        print("Finish Testing")


def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
        <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""

    colors = np.array([[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.]])

    bones = [((1, 0), colors[0, :]),
             ((2, 1), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((4, 3), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)


def plt_3d_hand(estimated_uvd, ax2):
    skeletonss = estimated_uvd.copy()
    for i in range(21):
        ax2.scatter(xs=skeletonss[i, 0],
                    ys=skeletonss[i, 1],
                    zs=skeletonss[i, 2], c='r', marker='o')

    ax2.plot([skeletonss[0, 0], skeletonss[1, 0]],
             [skeletonss[0, 1], skeletonss[1, 1]],
             zs=[skeletonss[ 0, 2], skeletonss[1, 2]], c='blue')
    ax2.plot([skeletonss[1, 0], skeletonss[2, 0]],
             [skeletonss[1, 1], skeletonss[2, 1]],
             zs=[skeletonss[1, 2], skeletonss[2, 2]], c='blue')
    ax2.plot([skeletonss[2, 0], skeletonss[3, 0]],
             [skeletonss[2, 1], skeletonss[3, 1]],
             zs=[skeletonss[2, 2], skeletonss[3, 2]], c='blue')
    ax2.plot([skeletonss[3, 0], skeletonss[4, 0]],
             [skeletonss[3, 1], skeletonss[4, 1]],
             zs=[skeletonss[3, 2], skeletonss[4, 2]], c='blue')

    ax2.plot([skeletonss[0, 0], skeletonss[5, 0]],
             [skeletonss[0, 1], skeletonss[5, 1]],
             zs=[skeletonss[0, 2], skeletonss[5, 2]], c='magenta')
    ax2.plot([skeletonss[5, 0], skeletonss[6, 0]],
             [skeletonss[5, 1], skeletonss[6, 1]],
             zs=[skeletonss[5, 2], skeletonss[6, 2]], c='magenta')
    ax2.plot([skeletonss[6, 0], skeletonss[7, 0]],
             [skeletonss[6, 1], skeletonss[7, 1]],
             zs=[skeletonss[6, 2], skeletonss[7, 2]], c='magenta')
    ax2.plot([skeletonss[7, 0], skeletonss[8, 0]],
             [skeletonss[7, 1], skeletonss[8, 1]],
             zs=[skeletonss[7, 2], skeletonss[8, 2]], c='magenta')

    ax2.plot([skeletonss[0, 0], skeletonss[9, 0]],
             [skeletonss[0, 1], skeletonss[9, 1]],
             zs=[skeletonss[0, 2], skeletonss[9, 2]], c='green')
    ax2.plot([skeletonss[9, 0], skeletonss[10, 0]],
             [skeletonss[9, 1], skeletonss[10, 1]],
             zs=[skeletonss[9, 2], skeletonss[10, 2]], c='green')
    ax2.plot([skeletonss[10, 0], skeletonss[11, 0]],
             [skeletonss[10, 1], skeletonss[11, 1]],
             zs=[skeletonss[10, 2], skeletonss[11, 2]], c='green')
    ax2.plot([skeletonss[11, 0], skeletonss [12, 0]],
             [skeletonss[11, 1], skeletonss[12, 1]],
             zs=[skeletonss[11, 2], skeletonss[12, 2]], c='green')

    ax2.plot([skeletonss[0, 0], skeletonss[13, 0]],
             [skeletonss[0, 1], skeletonss[13, 1]],
             zs=[skeletonss[0, 2], skeletonss[13, 2]], c='black')
    ax2.plot([skeletonss[13, 0], skeletonss[14, 0]],
             [skeletonss[13, 1], skeletonss[14, 1]],
             zs=[skeletonss[13, 2], skeletonss[14, 2]], c='black')
    ax2.plot([skeletonss[14, 0], skeletonss[15, 0]],
             [skeletonss[14, 1], skeletonss[15, 1]],
             zs=[skeletonss[ 14, 2], skeletonss[15, 2]], c='black')
    ax2.plot([skeletonss[15, 0], skeletonss[16, 0]],
             [skeletonss[15, 1], skeletonss[16, 1]],
             zs=[skeletonss[15, 2], skeletonss[16, 2]], c='black')

    ax2.plot([skeletonss[0, 0], skeletonss[17, 0]],
             [skeletonss[0, 1], skeletonss[17, 1]],
             zs=[skeletonss[0, 2], skeletonss[17, 2]], c='red')
    ax2.plot([skeletonss[17, 0], skeletonss[18, 0]],
             [skeletonss[17, 1], skeletonss[18, 1]],
             zs=[skeletonss[17, 2], skeletonss[18, 2]], c='red')
    ax2.plot([skeletonss[18, 0], skeletonss[19, 0]],
             [skeletonss[18, 1], skeletonss[19, 1]],
             zs=[skeletonss[18, 2], skeletonss[19, 2]], c='red')
    ax2.plot([skeletonss[19, 0], skeletonss[20, 0]],
             [skeletonss[19, 1], skeletonss[20, 1]],
             zs=[skeletonss[19, 2], skeletonss[20, 2]], c='red')

    plt.title('Estimated 3D(UVD)')
    # plt.show()


def main():
    epochs = 100#100
    batchSize = 32 #2의제곱이 가장 best #32
    learningRate = 1e-5
    dataset_name = 'obman'
    num_workers = 8

   # trainer = Trainer(epochs, batchSize, learningRate, dataset_name, 'train', num_workers)
   # trainer.train()

    tester = Tester(batchSize, dataset_name, 'test', num_workers)
    tester.test()



if __name__ == '__main__':
    main()
#%%
