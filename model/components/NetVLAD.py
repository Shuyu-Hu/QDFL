import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from utils.commons import print_nb_params
# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        # x = x.unsqueeze(0)
        N, C, H, W= x.shape[:4]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        #[B,C,H,W]-->[B,K,H*W]
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)  # 1by1卷积，得到每个特征点对K个聚类中心的分量
        soft_assign = F.softmax(soft_assign, dim=1)   # 通过softmax得到百分比的形式

        # -----------------------------------------------------------------------
        # soft_assign_HW = soft_assign.view(N, self.num_clusters, H, W)
        # jz = torch.zeros([H, W], dtype=x.dtype, layout=x.layout, device=x.device)
        # for i in range(H):
        #     for j in range(W):
        #         if i < 15 and j < 15:
        #             if i <= j:
        #                 jz[i][j] = i
        #             else:
        #                 jz[i][j] = j
        #         if i < 15 and j >= 15:
        #             if i <= (29 - j):
        #                 jz[i][j] = i
        #             else:
        #                 jz[i][j] = (29 - j)
        #         if i >= 15 and j < 15:
        #             if j <= (29 - i):
        #                 jz[i][j] = j
        #             else:
        #                 jz[i][j] = (29 - i)
        #         if i >= 15 and j >= 15:
        #             if (29 - i) <= (29 - j):
        #                 jz[i][j] = (29 - i)
        #             else:
        #                 jz[i][j] = (29 - j)
        # jz = jz + 1
        # for n in range(N):
        #     for clu in range(self.num_clusters):
        #         soft_assign_HW[n][clu] = torch.mul(soft_assign_HW[n][clu], jz)
        # soft_assign = soft_assign_HW.view(N, self.num_clusters, -1)
        # ---------------------------------------------------------------

        # soft:[B,C,H,W]-->[B,K,H*W]; x: [B,C,H,W]-->[B,C,H*W]
        x_flatten = x.view(N, C, -1)


        # Vlad: [B,K,C]
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)

        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage
            # in loop
            # x: [B,C,H*W]-->[1,B,C,H*W]-->[B,1,C,H*W]
            # centroids: [1,C]-->[1,1,C,H*W]
            #该部分减法使用了broadcast
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        # [B,K,C]-->[B,K*C]
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)       # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


def main():
    out_dim = 1024
    x = torch.randn(5, out_dim, 12, 12)
    # x = torch.randn(2, out_dim, 32, 32)  # b,c,h,w
    agg = NetVLAD(num_clusters=8, dim=out_dim, normalize_input=True, vladv2=True).eval()
    print(agg)
    print_nb_params(agg)
    x_out = agg(x)

    from utils.commons import evaluate_model
    evaluate_model(agg, x, 0)
    print(x_out.shape)


if __name__ == '__main__':
    main()
