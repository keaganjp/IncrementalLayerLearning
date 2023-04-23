from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt



class ILLLinearLayerOD(nn.Linear):
    def __init__(self, in_features, out_features, num_classes, layer_lr = 0.001,
                 num_epochs = 25,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = SGD(self.parameters(), lr=layer_lr)
        self.num_epochs = num_epochs
        self.predictor = nn.Linear(out_features, num_classes, device=device)
        self.predictor.requires_grad = False
        self.regressor = nn.Linear(out_features, num_classes*4, device=device)
        self.regressor.requires_grad = False
    
    def forward(self, x):
        layerOutput = self.relu(super().forward(x))
        predictorOutput = self.predictor.forward(layerOutput)
        regressorOutput = self.regressor.forward(regressorOutput)
        return layerOutput, predictorOutput
    
    def predict(self, x):
        layerOutput, predictorOutput, regressorOutput = self.forward(x)
        return F.softmax(predictorOutput, dim=1)
    
    def trainLayer(self, dataloader, previousLayers):
        for epoch in range(self.num_epochs):
              criterion = nn.CrossEntropyLoss()
              for i, data in enumerate(dataloader):
                  originalInputs, labels = data
                  originalInputs = originalInputs.to(device)
                  labels = labels.to(device)
                  inputs = originalInputs
                  for previous in previousLayers:
                      if isinstance(previous, nn.MaxPool2d) or isinstance(previous, nn.Flatten):
                          inputs = previous.forward(inputs)
                      else:
                          inputs,_ = previous.forward(inputs)
                  self.opt.zero_grad()
                  layerOutput, predictorOutput = self.forward(inputs)
                  layerLoss = criterion(predictorOutput, labels)
                  # This is a local layer update, not a backprop through the net
                  layerLoss.backward()
                  self.opt.step()


# Reeze all layers - use classifier built from ILL layers
def ill_decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    
    classifier = nn.Sequential(nn.Linear(in_features=25088, out_features = 4096, bias = True),
                               nn.ReLU())
    # freeze all conv
    for layer in features:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier



# RPN stays the same - needs only one output set, so nothing changes. Just need to train it first
# Head is what changes - and that's just the classifier

class ILLFasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = ill_decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = ILLVGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(ILLFasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class ILLVGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(ILLVGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        #self.cls_loc.requires_grad = False
        #self.score.requires_grad= False

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
