import sys
sys.path.append('./SOLIDER-REID/')
import os
from config import cfg
import argparse
from model import make_model
import metrics
import torchvision.transforms as T
import torch

class CFG (object):
    pass
class REID:
    def __init__(self,config_file,test_weight_path):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.dist_metric = 'euclidean'
        self.args=CFG()
        self.cfg=cfg
        self.args.config_file=config_file

        self.cfg.merge_from_file(self.args.config_file)
        self.cfg.freeze()
        self.cfg.defrost()
        self.cfg.TEST.WEIGH=test_weight_path
        #load model
        self.model=self._build_model()
        # load transforms
        self.transforms=self.__load_transforms()
    def _build_model(self):
        model = make_model(cfg, num_class=1, camera_num=None, view_num =None, semantic_weight = self.cfg.MODEL.SEMANTIC_WEIGHT)
        model.to(self.device)
        model.load_param(self.cfg.TEST.WEIGH)
        model.eval()
        return model

    def __load_transforms(self):
        transforms= T.Compose([
        T.ToTensor(),
        T.Resize(self.cfg.INPUT.SIZE_TEST),
        T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)])

        return transforms
    def _preprocess_data(self,img):
        img=self.transforms(img)
        img=img.unsqueeze(0)
        img=img.to(self.device)
        return img

    @torch.no_grad()
    def _extract_features(self, inputs):
        return self.model(inputs, cam_label= None, view_label=None)

    def _features(self, imgs):
        f = []
        for img in imgs:
            img=self._preprocess_data(img)
            features= self._extract_features(img)[0].cpu()
            f.append(features)
        f = torch.cat(f, 0)
        return f
    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        # print(distmat.shape)
        return distmat.numpy()


if __name__ == '__main__':
    reid = REID()