from torch import nn


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, base_width=64):
        super().__init__()
        norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = norm_layer(planes)

        if inplanes != planes:
            self.identity_fn = nn.Sequential(
                nn.Conv1d(inplanes, planes * self.expansion, 1),
                norm_layer(planes * self.expansion),
            )
        else:
            self.identity_fn = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.identity_fn(identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self, inplanes, planes, base_width=64):
        super().__init__()
        norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(inplanes, width, 1)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv1d(width, planes * self.expansion, 1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if inplanes != planes:
            self.identity_fn = nn.Sequential(
                nn.Conv1d(inplanes, planes * self.expansion, 1),
                norm_layer(planes * self.expansion),
            )
        else:
            self.identity_fn = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.identity_fn(identity)
        out = self.relu(out)

        return out


class ResNetDecoder(nn.Module):

    def __init__(self, in_feats, num_tokens, block, layers, zero_init_residual=False, width_per_group=64, ce_output=True, bbox=False, sem_head=False, geo_head=False, arti_head=False, pc_head=False, ft_category=None, pred_junc=False, corners=False):
        super().__init__()
        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.num_tokens = num_tokens
        self.inplanes = 512
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_feats, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 512, layers[0])
        self.layer2 = self._make_layer(block, 384, layers[1])
        self.layer3 = self._make_layer(block, 384, layers[2])
        self.layer4 = self._make_layer(block, 320, layers[3])
        self.ce_output = ce_output
        self.bbox = bbox
        self.sem_head = sem_head
        self.geo_head = geo_head
        self.arti_head = arti_head
        self.pc_head = pc_head
        self.ft_category = ft_category
        self.pred_junc = pred_junc
        self.corners = corners

        if not bbox:
            if ce_output:
                self.fc = nn.Conv1d(320 * block.expansion, self.num_tokens * 9, 1)
                if self.sem_head:
                    if self.ft_category is None: # if mix category, regress the class category text feature embedding, then retrieve the label
                        self.fc_semantic_head = nn.Conv1d(320 * block.expansion, 768, 1)
                        
                if self.geo_head:
                    self.fc_geo_head = nn.Conv1d(320 * block.expansion, 128, 1)
                if self.arti_head:
                    self.fc_arti_exist_head = nn.Conv1d(320 * block.expansion, 2, 1)
                    self.fc_arti_loc_head = nn.Conv1d(320 * block.expansion, self.num_tokens * 3, 1)
                    self.fc_arti_ori_head = nn.Conv1d(320 * block.expansion, self.num_tokens * 3, 1)
                    self.fc_arti_type_head = nn.Conv1d(320 * block.expansion, 3, 1)
                if self.pc_head:
                    self.fc_pc_head = nn.Conv1d(320 * block.expansion, 2 * 9, 1)
                if self.pred_junc:
                    self.fc_junc = nn.Conv1d(320 * block.expansion, 1, 1)
            else:
                self.fc = nn.Conv1d(320 * block.expansion, 9, 1)
        else:
            if ce_output:
                if corners:
                    self.fc = nn.Conv1d(320 * block.expansion, self.num_tokens * 24, 1)
                else:
                    self.fc = nn.Conv1d(320 * block.expansion, self.num_tokens * 6, 1)
                if self.sem_head:
                    self.fc_semantic_head = nn.Conv1d(320 * block.expansion, 768, 1)
                if self.geo_head:
                    self.fc_geo_head = nn.Conv1d(320 * block.expansion, 128, 1)
                if self.arti_head:
                    self.fc_arti_exist_head = nn.Conv1d(320 * block.expansion, 2, 1)
                    self.fc_arti_loc_head = nn.Conv1d(320 * block.expansion, self.num_tokens * 3, 1)
                    self.fc_arti_ori_head = nn.Conv1d(320 * block.expansion, self.num_tokens * 3, 1)
                    self.fc_arti_type_head = nn.Conv1d(320 * block.expansion, 3, 1)
            else:
                self.fc = nn.Conv1d(320 * block.expansion, 6, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes: int, blocks: int):
        layers = [block(self.inplanes, planes, self.base_width)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        B, _, N = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.sem_head:
            x_sem = self.fc_semantic_head(x)
        if self.geo_head:
            x_geo = self.fc_geo_head(x)
        if self.arti_head:
            x_arti_exist = self.fc_arti_exist_head(x)
            x_arti_loc = self.fc_arti_loc_head(x)
            x_arti_ori = self.fc_arti_ori_head(x)
            x_arti_type = self.fc_arti_type_head(x)
        if self.pc_head:
            x_pc = self.fc_pc_head(x)
        if self.pred_junc:
            x_junc = self.fc_junc(x)
        x = self.fc(x)
        if not self.bbox:
            if self.ce_output:
                x = x.permute((0, 2, 1)).reshape(B, N, 9, self.num_tokens)
                if self.sem_head:
                    if self.ft_category is None: # mix category, regression
                        x_sem = x_sem.permute((0, 2, 1))
                if self.geo_head:
                    x_geo = x_geo.permute((0, 2, 1))
                if self.arti_head:
                    x_arti_exist = x_arti_exist.permute((0, 2, 1)).reshape(B, N, 1, 2)
                    x_arti_type = x_arti_type.permute((0, 2, 1)).reshape(B, N, 1, 3)
                    x_arti_loc = x_arti_loc.permute((0, 2, 1)).reshape(B, N, 3, self.num_tokens)
                    x_arti_ori = x_arti_ori.permute((0, 2, 1)).reshape(B, N, 3, self.num_tokens)
                if self.pc_head:
                    x_pc = x_pc.permute((0, 2, 1))
                if self.pred_junc:
                    x_junc = x_junc.permute((0, 2, 1)).reshape(B, N, 1, 1)
            else:
                x = x.permute((0, 2, 1)).reshape(B, N, 9)
        else:
            if self.ce_output:
                if self.corners:
                    x = x.permute((0, 2, 1)).reshape(B, N, 24, self.num_tokens)
                else:
                    x = x.permute((0, 2, 1)).reshape(B, N, 6, self.num_tokens)
                if self.sem_head:
                    x_sem = x_sem.permute((0, 2, 1))
                if self.geo_head:
                    x_geo = x_geo.permute((0, 2, 1))
                if self.arti_head:
                    x_arti_exist = x_arti_exist.permute((0, 2, 1)).reshape(B, N, 1, 2)
                    x_arti_type = x_arti_type.permute((0, 2, 1)).reshape(B, N, 1, 3)
                    x_arti_loc = x_arti_loc.permute((0, 2, 1)).reshape(B, N, 3, self.num_tokens)
                    x_arti_ori = x_arti_ori.permute((0, 2, 1)).reshape(B, N, 3, self.num_tokens)
            else:
                x = x.permute((0, 2, 1)).reshape(B, N, 6)
        if self.sem_head:
            if self.geo_head and self.arti_head:
                return x, x_sem, x_geo, x_arti_exist, x_arti_loc, x_arti_ori, x_arti_type
            elif self.arti_head and self.pc_head:
                return x, x_sem, x_pc, x_arti_exist, x_arti_loc, x_arti_ori, x_arti_type
            elif self.geo_head and self.pc_head:
                return x, x_sem, x_geo, x_pc
            elif self.geo_head:
                return x, x_sem, x_geo
            elif self.pc_head:
                return x, x_sem, x_pc
            elif self.arti_head:
                return x, x_sem, x_arti_exist, x_arti_loc, x_arti_ori, x_arti_type
            return x, x_sem
        elif self.pred_junc:
            return x, x_junc
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetEncoder(nn.Module):

    def __init__(self, in_feats, block, layers, zero_init_residual=False, width_per_group=64):
        super().__init__()
        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.inplanes = 128
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_feats, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 192, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 384, layers[3])

        self.fc = nn.Conv1d(384 * block.expansion, 512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes: int, blocks: int):
        layers = [block(self.inplanes, planes, self.base_width)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        B, C, N = x.shape
        x = x.permute((0, 2, 1)).reshape(-1, C)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18_decoder(in_feats, num_tokens, ce_output=True, sem_head=False, pc_head=False):
    return ResNetDecoder(in_feats, num_tokens, BasicBlock, [2, 2, 2, 2], zero_init_residual=True, ce_output=ce_output, sem_head=sem_head, pc_head=pc_head)

def resnet34_decoder(in_feats, num_tokens, ce_output=True, sem_head=False, geo_head=False, arti_head=False, pc_head=False, ft_category=None, pred_junc=False):
    return ResNetDecoder(in_feats, num_tokens, BasicBlock, [3, 4, 6, 3], zero_init_residual=True, ce_output=ce_output, sem_head=sem_head, geo_head=geo_head, arti_head=arti_head, pc_head=pc_head, ft_category=ft_category, pred_junc=pred_junc)

def resnet18_encoder(in_feats):
    return ResNetEncoder(in_feats, BasicBlock, [2, 2, 2, 2], zero_init_residual=True)

def resnet34_encoder(in_feats):
    return ResNetEncoder(in_feats, BasicBlock, [3, 4, 6, 3], zero_init_residual=True)

def test_resnet_decoder():
    import torch
    model = resnet18_decoder(512, 65)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

    model = model.cuda()
    x_ = torch.rand(1, 512, 2048).cuda()
    y_, c_ = model(x_)
    print(y_.shape, c_.shape)


def test_resnet_encoder():
    import torch
    model = resnet18_encoder(70)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

    model = model.cuda()
    x_ = torch.rand(1, 70, 2048).cuda()
    y_ = model(x_)
    print(y_.shape)


if __name__ == "__main__":
    test_resnet_encoder()
