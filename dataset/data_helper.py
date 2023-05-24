
import os
import json
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.BASE_DIR = args.base_dir
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    def _clean_report(self, report):
        report = report.replace('\n', '').replace('_', '').strip().lower().split('.')
        return report.strip()


    def parse(self, features, training=False):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        text1 = self._clean_report(report)
        to_return['input_text'] = text1
        # chest x-ray images
        try:
            with Image.open(os.path.join(self.BASE_DIR, features['image_path'][0])) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                features['image'] = array
                to_return["image_mask"] = 0
        except Exception as e:
            print('can not find image')
            features['image'] = np.zeros((self.args.image_width, self.args.image_height, 3), dtype=np.uint8)
            to_return["image_mask"] = 1

        to_return["image"] = self._parse_image(features['image'])
        return to_return


    def transform_with_parse(self, inputs, training=True):
        return self.parse(inputs, training)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)
        self.training = True if split == 'train' else False

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index], self.training)


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    data = json.load(open('../data/mimic_annotation.json', 'r'))


