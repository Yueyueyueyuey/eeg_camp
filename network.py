import torch.nn as nn
import torch

class EEG_Denoise(nn.Module):

    def __init__(self):
        super(EEG_Denoise, self).__init__()

        self.denoise_l1_n1_v1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l1_n1_v2 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l1_n1_v3 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l1_n1_v4 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l1_n2_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l1_n2_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l1_n2_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l1_n2_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l1_n3_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l1_n3_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l1_n3_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l1_n3_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l1_n4_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l1_n4_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l1_n4_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l1_n4_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv1_squeeze1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)

        self.denoise_l2_n1_v1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l2_n1_v2 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l2_n1_v3 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l2_n1_v4 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l2_n2_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l2_n2_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l2_n2_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l2_n2_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l2_n3_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l2_n3_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l2_n3_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l2_n3_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l2_n4_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l2_n4_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l2_n4_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l2_n4_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv2_squeeze1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)

        self.denoise_l3_n1_v1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l3_n1_v2 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l3_n1_v3 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l3_n1_v4 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l3_n2_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l3_n2_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l3_n2_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l3_n2_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l3_n3_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l3_n3_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l3_n3_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l3_n3_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.denoise_l3_n4_v1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.denoise_l3_n4_v2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.denoise_l3_n4_v3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.denoise_l3_n4_v4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv3_squeeze1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)

        self.batch_norm = nn.BatchNorm1d(512, affine=True)

    def forward(self, x):
        denoise_f, denoise_s = [x, x]

        denoise_f = torch.unsqueeze(denoise_f, 1)

        denoise_1 = self.denoise_l1_n1_v1(denoise_f)
        denoise_2 = self.denoise_l1_n1_v2(denoise_f)
        denoise_3 = self.denoise_l1_n1_v3(denoise_f)
        denoise_4 = self.denoise_l1_n1_v4(denoise_f)

        denoise_f = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_f = torch.relu(denoise_f)
        denoise_skip_connect_x = denoise_f

        denoise_1 = self.denoise_l1_n2_v1(denoise_f)
        denoise_2 = self.denoise_l1_n2_v2(denoise_f)
        denoise_3 = self.denoise_l1_n2_v3(denoise_f)
        denoise_4 = self.denoise_l1_n2_v4(denoise_f)
        denoise_f = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_f = torch.sigmoid(denoise_f)

        denoise_1 = self.denoise_l1_n3_v1(denoise_f)
        denoise_2 = self.denoise_l1_n3_v2(denoise_f)
        denoise_3 = self.denoise_l1_n3_v3(denoise_f)
        denoise_4 = self.denoise_l1_n3_v4(denoise_f)
        denoise_f = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_f = torch.sigmoid(denoise_f)
        denoise_f = denoise_f + denoise_skip_connect_x

        denoise_1 = self.denoise_l1_n4_v1(denoise_f)
        denoise_2 = self.denoise_l1_n4_v2(denoise_f)
        denoise_3 = self.denoise_l1_n4_v3(denoise_f)
        denoise_4 = self.denoise_l1_n4_v4(denoise_f)
        denoise_f = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)

        denoise_f = self.conv1_squeeze1(denoise_f)

        denoise_f = torch.squeeze(denoise_f, 1)

        denoise_s = torch.unsqueeze(denoise_s, 1)

        denoise_1 = self.denoise_l2_n1_v1(denoise_s)
        denoise_2 = self.denoise_l2_n1_v2(denoise_s)
        denoise_3 = self.denoise_l2_n1_v3(denoise_s)
        denoise_4 = self.denoise_l2_n1_v4(denoise_s)

        denoise_s = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_s = torch.relu(denoise_s)
        denoise_skip_connect_x = denoise_s

        denoise_1 = self.denoise_l2_n2_v1(denoise_s)
        denoise_2 = self.denoise_l2_n2_v2(denoise_s)
        denoise_3 = self.denoise_l2_n2_v3(denoise_s)
        denoise_4 = self.denoise_l2_n2_v4(denoise_s)
        denoise_s = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_s = torch.sigmoid(denoise_s)

        denoise_1 = self.denoise_l2_n3_v1(denoise_s)
        denoise_2 = self.denoise_l2_n3_v2(denoise_s)
        denoise_3 = self.denoise_l2_n3_v3(denoise_s)
        denoise_4 = self.denoise_l2_n3_v4(denoise_s)
        denoise_s = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_s = torch.sigmoid(denoise_s)
        denoise_s = denoise_s + denoise_skip_connect_x

        denoise_1 = self.denoise_l2_n4_v1(denoise_s)
        denoise_2 = self.denoise_l2_n4_v2(denoise_s)
        denoise_3 = self.denoise_l2_n4_v3(denoise_s)
        denoise_4 = self.denoise_l2_n4_v4(denoise_s)
        denoise_s = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)

        denoise_s = self.conv2_squeeze1(denoise_s)
        denoise_s = torch.sigmoid(denoise_s)
        denoise_s = torch.squeeze(denoise_s, 1)

        denoise_end = torch.mul(denoise_f, denoise_s)

        denoise_end = torch.unsqueeze(denoise_end, 1)

        denoise_1 = self.denoise_l3_n1_v1(denoise_end)
        denoise_2 = self.denoise_l3_n1_v2(denoise_end)
        denoise_3 = self.denoise_l3_n1_v3(denoise_end)
        denoise_4 = self.denoise_l3_n1_v4(denoise_end)

        denoise_end = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_end = torch.relu(denoise_end)
        denoise_skip_connect_x = denoise_end

        denoise_1 = self.denoise_l3_n2_v1(denoise_end)
        denoise_2 = self.denoise_l3_n2_v2(denoise_end)
        denoise_3 = self.denoise_l3_n2_v3(denoise_end)
        denoise_4 = self.denoise_l3_n2_v4(denoise_end)
        denoise_end = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_end = torch.sigmoid(denoise_end)

        denoise_1 = self.denoise_l3_n3_v1(denoise_end)
        denoise_2 = self.denoise_l3_n3_v2(denoise_end)
        denoise_3 = self.denoise_l3_n3_v3(denoise_end)
        denoise_4 = self.denoise_l3_n3_v4(denoise_end)
        denoise_end = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_end = torch.sigmoid(denoise_end)
        denoise_end = denoise_end + denoise_skip_connect_x

        denoise_1 = self.denoise_l3_n4_v1(denoise_end)
        denoise_2 = self.denoise_l3_n4_v2(denoise_end)
        denoise_3 = self.denoise_l3_n4_v3(denoise_end)
        denoise_4 = self.denoise_l3_n4_v4(denoise_end)
        denoise_end = torch.cat((denoise_1, denoise_2, denoise_3, denoise_4), dim=1)
        denoise_end = self.conv3_squeeze1(denoise_end)

        denoise_end = torch.squeeze(denoise_end, 1)

        return denoise_end