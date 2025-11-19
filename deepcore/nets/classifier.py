'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.nets_utils import EmbeddingRecorder   # <--- IMPORTANTE


class ModelM7_small(nn.Module):
    def __init__(self, record_embedding=False, no_grad=False):
        super(ModelM7_small, self).__init__()

        # ==== TUS CAPAS ====
        self.conv1 = nn.Conv2d(1, 24, 7, bias=False)
        self.conv1_bn = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 48, 7, bias=False)
        self.conv2_bn = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(48, 72, 7, bias=False)
        self.conv3_bn = nn.BatchNorm2d(72)

        self.conv4 = nn.Conv2d(72, 96, 7, bias=False)
        self.conv4_bn = nn.BatchNorm2d(96)

        self.fc1 = nn.Linear(1536, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

        # ==== EXTRA PARA DEEPCORE ====
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    # === MUY IMPORTANTE (DeepCore lo usa para obtener gradientes) ===
    def get_last_layer(self):
        return self.fc1

    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))

        flat = torch.flatten(conv4.permute(0, 2, 3, 1), 1)

        # DeepCore puede grabar esto si lo habilitas
        flat = self.embedding_recorder(flat)

        logits = self.fc1_bn(self.fc1(flat))
        return logits

    def forward(self, x):
        with torch.set_grad_enabled(not self.no_grad):
            logits = self.get_logits(x)
        return logits
'''