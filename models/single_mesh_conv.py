from custom_types import *
import torch.nn as nn


class MultiMeshConv(nn.Module):

    def __init__(self, number_features: Union[Tuple[int, ...], List[int]]):
        super(MultiMeshConv, self).__init__()
        layers = [
          nn.Sequential(*
                        [SingleMeshConv(number_features[i], number_features[i + 1], i==0)] +
                        ([nn.InstanceNorm1d(number_features[i + 1])]) +
                        [nn.LeakyReLU(0.2, inplace=True)]
          ) for i in range(len(number_features) - 2)
        ] + [SingleMeshConv(number_features[-2], number_features[-1], False)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, gfmm):
        for layer in self.layers:
            x = layer((x, gfmm))
        return x

class SingleMeshConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, is_first):
        super(SingleMeshConv, self).__init__()
        self.first = is_first
        if is_first:
            self.conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.conv = nn.Conv1d(in_channels * 2, out_channels, 1)

    def forward(self, mesh: TS) -> T:
        # From MeshHandler's extract_features, x [1, 4 features, faces]
        x, gfmm = mesh
        n_faces = x.shape[-1]  # 1, in_fe, f
        if not self.first:
            # x_a [1, 4 features, 3 neighbor faces, faces]
            x_a = x[:, :, gfmm]
            # x_b [1, 4 features, 3 neighbor faces, faces] self feature dup 3 times
            x_b = x.view(1, -1, 1, n_faces).expand_as(x_a)
            # x [1, 8 features, 3 neighbor faces, faces]
            x = torch.cat((x_a, x_b), 1)
        else:
            # create initial feature's embedding
            x = x.view(1, 3, -1, n_faces).permute(0, 2, 1, 3)
        # x [1, 8 features, 3 neighbor faces * faces]
        x = x.reshape(1, -1, n_faces * 3)
        # x [1, 8 or 4 features, 3 neighbor faces * faces]
        x = self.conv(x)
        # x [1, 8 or 4 features, 3 neighbor faces, faces]
        x = x.view(1, -1, 3, n_faces)
        # Prevent from face order
        # x [1, 8 or 4 features, faces]
        x = x.max(2)[0]
        return x