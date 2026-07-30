import torch
import numpy as np
import torch.nn.functional as F

class SO2_rotz(object):
    def __init__(self, rot_range):
        # rot_range: [min, max] e.g.[-pi, pi]
        self.rot_range = rot_range

    def check_numpy_to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float(), True
        return x, False

    def rotate_points_along_z(self, points, angle):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:

        """
        points, is_numpy = self.check_numpy_to_torch(points)
        angle, _ = self.check_numpy_to_torch(angle)

        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot.numpy() if is_numpy else points_rot
    
    def __call__(self, points, gt_bboxes, eval=False, radian=None):
        '''
            data: input point cloud data with batchsize=1, (N, 3 + C_in)
            gt_bboxes: [N, 7] (x, y ,z, w, h, l, r_y)

            return: norm: normal vector of each point, (N, 3)
        '''
        
        if radian is None:
            noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
            noise_rotation = 0.
        else:
            noise_rotation = radian
            
        if eval is False:
            points = self.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
#             print(noise_rotation)
            if gt_bboxes is not None:
                gt_bboxes[:, 0:3] = self.rotate_points_along_z(gt_bboxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
                gt_bboxes[:, 6] += noise_rotation
                gt_bboxes[:, 6] %= (2*np.pi)
                
            return gt_bboxes, points, noise_rotation
                # if gt_bboxes.shape[1] > 7:
                #     gt_bboxes[:, 7:9] = self.rotate_points_along_z(
                #         np.hstack((gt_bboxes[:, 7:9], np.zeros((gt_bboxes.shape[0], 1))))[np.newaxis, :, :],
                #         np.array([noise_rotation])
                #     )[0][:, 0:2]
        else:
            gt_bboxes[:, 0:3] = self.rotate_points_along_z(gt_bboxes[np.newaxis, :, 0:3], np.array([-noise_rotation]))[0]
            gt_bboxes[:, 6] -= noise_rotation
            gt_bboxes[:, 6] %= (2*np.pi)
            
            return gt_bboxes, points
