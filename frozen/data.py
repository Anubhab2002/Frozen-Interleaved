import os
import os.path as osp
import json
import logging

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Lambda
from transformers import GPT2Tokenizer, AutoFeatureExtractor, CLIPFeatureExtractor
from pytorch_lightning import LightningDataModule
from timm.data.transforms_factory import create_transform

from .util import is_clip_model


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_TOKEN = "<image>"
SPECIAL_TOKEN_DICT = {'additional_special_tokens': [IMAGE_TOKEN]}
NUM_IMAGE_TOKENS = 2
PAD_TOKEN_ID = 1


TIMM_CONFIGS = {
    'nf_resnet50':  {
        'input_size': (3, 256, 256),
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'crop_pct': 0.94,
    },
}


def get_image_transform(model_name):
    if model_name in TIMM_CONFIGS.keys():
        config = TIMM_CONFIGS[model_name]
        transform = create_transform(**config)
        transform.transforms.append(
            Lambda(lambda x: x.unsqueeze(0)),
        )
    elif is_clip_model(model_name):
        transform = CLIPFeatureExtractor.from_pretrained(model_name)
    else:
        transform = AutoFeatureExtractor.from_pretrained(model_name)
    return transform


class COCODataset(Dataset):
    def __init__(
        self,
        name='COCO',
        path=None,
        split='val',
        year=2017,
        image_transform=None,
        tokenizer=None,
        num_image_tokens=0,
    ):
        super().__init__()
        assert split in ('train', 'val')
        assert year in (2014, 2017)
        logging.warn(f'num_image_tokens = {num_image_tokens}')

        self.path = osp.abspath(osp.expanduser(path))
        self.split = split
        self.year = year
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens

        if self.tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-2.7b")

        if not IMAGE_TOKEN in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens(SPECIAL_TOKEN_DICT)

        self.setup_dataset()

    def setup_dataset(self):
        split, year = self.split, self.year
        self.split_name = f'{split}{year}'
        self.image_dir = osp.join(self.path, self.split_name)
        self.annotation_file = osp.join(
            self.path, 'annotations', f'captions_{self.split_name}.json')

        with open(self.annotation_file, 'r') as f:
            json_data = json.load(f)
            annotations = json_data['annotations']

        image_dict = dict()
        for item in json_data['images']:
            image_dict[item['id']] = item

        self.annotations = annotations
        self.image_dict = image_dict

    @property
    def image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def __len__(self):
        return len(self.annotations)

    def _read_image(self, index):
        image_id = self.annotations[index]['image_id']
        file_name = self.image_dict[image_id]['file_name']
        file_path = osp.join(self.image_dir, file_name)
        raw = Image.open(file_path)
        raw = raw.convert('RGB') if raw.mode != 'RGB' else raw
        if isinstance(self.image_transform, Compose):
            image = self.image_transform(raw)
        elif self.image_transform is not None:  # HuggingFace
            image = self.image_transform(raw, return_tensors='pt')
            image = image['pixel_values']
        return image_id, raw, image

    def _add_image_tokens(self, caption):
        N = self.num_image_tokens
        if N is not None or N > 0:
            tokens = ' '.join([IMAGE_TOKEN for x in range(N)])
            caption = f'{tokens} {caption}'
        return caption

    def _new_add_image_tokens(self, inputs):
        N = self.num_image_tokens
        I = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if N is not None or N > 0:
            input_ids = [I for i in range(N)] + inputs['input_ids']
            attention_mask = [1 for i in range(N)] + inputs['attention_mask']
        return {
            'input_ids': torch.tensor(input_ids).unsqueeze(0),
            'attention_mask': torch.tensor(attention_mask).unsqueeze(0),
        }

    def __getitem__(self, index):
        try:
            image_id, raw, image = self._read_image(index)
        except:
            image_id, raw, image = -1, None, None
        caption = self.annotations[index]['caption']
        inputs = self.tokenizer(caption)
        inputs = self._new_add_image_tokens(inputs)

        image_token_mask = inputs['input_ids'] == self.image_token_id

        return {
            'pixel_values': image,
            'caption': caption,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'image_token_mask': image_token_mask.long(),
            'item_id': index,
            'image_id': image_id,
            'caption_id': self.annotations[index]['id'],
            'raw_image': raw,
        }


class CC3MDataset(COCODataset):
    def setup_dataset(self):
        assert self.split in ('train', 'val')
        script_path = osp.join(self.path, 'script/DownloadConceptualCaptions/')
        annotation_file = 'sample_train.tsv'
        if self.split == 'val':
            annotation_file = 'sample_validation.tsv'
        annotations = pd.read_csv(
            osp.join(script_path, annotation_file),
            sep='\t',
            names=['caption', 'image_url'],
        )

        split_ = 'training' if self.split == 'train' else 'validation'
        download_file = osp.join(script_path, f'downloaded_{split_}_report.tsv')
        download_report = pd.read_csv(
            osp.join(script_path, download_file),
            sep='\t',
            # names=["image_url", "split", "status", "file_name"]
            names=["file_name", "split", "type", "size", "status", "image_url"])
        download_report = download_report[download_report['status'] == 200]
        # print("Images downloaded: ",  download_report.shape)
        download_report = download_report.loc[:, ['image_url', 'file_name']]
        # print(download_report.head())
        records = annotations.merge(download_report)
        self.annotations = records.to_dict('records')
        # print("*********TRAIN ANNOTATION*******\n", self.annotations)
        self.image_dir = osp.join(self.path, split_)
        # print(self.image_dir)

    def _read_image(self, index):
        file_name = self.annotations[index]['file_name']
        file_path = osp.join(self.path, 'script/DownloadConceptualCaptions', file_name)
        # print("File path: ", file_path)
        raw = Image.open(file_path)
        try:
            raw = Image.open(file_path)
            raw = raw.convert('RGB') if raw.mode != 'RGB' else raw
        except (UnidentifiedImageError, Image.DecompressionBombError) as e:
            raw = None
        # print("################RAW IMAGE:", raw)
        if raw is not None and isinstance(self.image_transform, Compose):
            image = self.image_transform(raw)
        elif raw is not None and self.image_transform is not None: # HuggingFace
            image = self.image_transform(raw, return_tensors='pt')
            image = image['pixel_values']
        return raw, image

    def __getitem__(self, index):
        item = self.annotations[index]
        try:
            raw, image = self._read_image(index)
        except:
            raw, image = None, None
        caption = item['caption']
        inputs = self.tokenizer(caption)
        inputs = self._new_add_image_tokens(inputs)
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        image_token_mask = inputs['input_ids'] == image_token_id

        return {
            'pixel_values': image,
            'caption': caption,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'image_token_mask': image_token_mask.long(),
            'item_id': index,
            'image_id': -1,
            'caption_id': -1,
            'raw_image': raw,
        }


class CaptioningDataModule(LightningDataModule):
    def __init__(
        self,
        config=dict(),
    ):
        super().__init__()
        self.config = config
        self.init_tokenizer()
        self.init_image_transform()
        self.load_splits()

    @property
    def loader_config(self):
        default_config = {
            'num_workers': 0,
            'pin_memory': False,
            'batch_size': 16,
        }
        return self.config.get('loader', default_config)

    @property
    def dataset_config(self):
        return self.config.get('dataset', dict())

    @property
    def model_config(self):
        return self.config.get('model', dict())

    def init_tokenizer(self):
        arch = self.model_config.get('text_encoder', 'facebook/opt-2.7b')
        self.tokenizer = GPT2Tokenizer.from_pretrained(arch)

    def init_image_transform(self):
        arch = self.model_config.get('image_encoder', 'microsoft/resnet-50')
        self.image_transform = get_image_transform(arch)

    def load_splits(self):
        self.train_data = self.load_split('train')
        # print("**** TRAIN DATA ****: ", self.train_data.__getitem__(0))
        self.val_data = self.load_split('val')

    def load_split(self, split):
        # print("*************************8Called with split: ", split)
        N = self.model_config.get('num_imagrane_tokens', NUM_IMAGE_TOKENS)
        dataset = self.dataset_config.get('name', 'COCO')
        # print("Dataset used: ", dataset)
        if dataset == 'COCO':
            dataset_class = COCODataset
        elif dataset == 'CC3M':
            dataset_class = CC3MDataset

        return dataset_class(
            split=split,
            tokenizer=self.tokenizer,
            image_transform=self.image_transform,
            **self.dataset_config,
        )

    def train_dataloader(self):
        print("Train dataloader function is called...")
        return DataLoader(
            self.train_data,
            collate_fn=collate_fn,
            shuffle=True,
            **self.loader_config,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            collate_fn=collate_fn,
            shuffle=False,
            **self.loader_config
        )

    def predict_dataloader(self):
        return self.val_dataloader()


def collate_fn(batch):
    # print("****************Batch: ", batch)
    batch = [x for x in batch if x['pixel_values'] is not None]
    batch_size = len(batch)
    longest = max([x['input_ids'].numel() for x in batch])
    pixel_values = torch.cat([x['pixel_values'] for x in batch])

    def init_helper(value, dtype):
        array = torch.empty((batch_size, longest), dtype=dtype)
        array.fill_(value)
        return array

    input_ids = init_helper(PAD_TOKEN_ID, torch.long)
    attention_mask = init_helper(0, torch.long)
    image_token_mask = init_helper(False, torch.long)

    for i in range(batch_size):
        length = batch[i]['input_ids'].numel()
        input_ids[i, :length] = batch[i]['input_ids']
        attention_mask[i, :length] = batch[i]['attention_mask']
        image_token_mask[i, :length] = batch[i]['image_token_mask']

    print("Pixel values device: ", pixel_values.device)
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image_token_mask': image_token_mask,
        'item_ids': [x['item_id'] for x in batch],
        'captions': [x['caption'] for x in batch],
        'image_ids': [x['image_id'] for x in batch],
        'caption_ids': [x['caption_id'] for x in batch],
    }

####################################
import pickle
import hashlib
from typing import Dict, List, Tuple
from abc import abstractmethod
from datasets import Dataset
from torch.utils.data import Dataset as torch_dataset

import pandas as pd
import numpy as np

from tqdm import tqdm

import ast

# returns the hexadecimal hash of the dataset
def hash_dataset(dataset):
    dataset_bytes = pickle.dumps(dataset)
    hash_obj = hashlib.sha256()
    hash_obj.update(dataset_bytes)
    return hash_obj.hexdigest()

class Utterance:
    text: str
    images: List[str]
    speaker: str | None

    def __init__(self, text: str, speaker: str | None = None):
        self.text = text
        self.speaker = speaker
        self.images = []
    
    # split the utterance at all possible breaking point
    def split(self):
        l = len(self.text)
        splits = []
        for i in range(0, l+1):
            utr = Utterance(self.text[:i], self.speaker)
            utr.images = self.images
            splits.append((utr, self.text[i:]))
        return splits
    
    def add_image(self, image: str):
        self.images.append(image)

    def format_utterance(self) -> str:
        formatted_utterance = "".join(
            ["[" + image + "]" for image in self.images] +
            [self.text]
        )
        return formatted_utterance

class Dialog:
    idx: str
    utterances: List[Utterance]

    def __init__(self, utterances: List[Utterance] | str, idx: str):
        if isinstance(utterances, str):
            self.utterances = self.get_utterances(utterances) # format str utterances to class format
        else:
            self.utterances = utterances
        self.idx = idx 

    def __getitem__(self, item: int) -> Utterance:
        return self.utterances[item]

    def __len__(self) -> int:
        return len(self.utterances)

    def __iter__(self):
        return iter(self.utterances)

    def format_dialog(self) -> str:
        formatted_dialog = "\n".join(
            [self.idx] + 
            [
                f"{utterance.speaker}: {utterance.format_utterance()}"
                for utterance in self.utterances
            ]
        )   
        return formatted_dialog

    def unroll(self):
        return [
            Dialog(self.utterances[:i], self.idx + "__u" + str(i))
            for i in range(1, len(self.utterances) + 1)
        ]
    
    def create_splits(self):
        context = self.utterances[:-1]
        response = self.utterances[-1]
        response_splits = response.split()
        return [
            (
                Dialog(
                    context + [response_split[0]], self.idx + "__s" + str(i)
                ),
                response_split[1]
            )
            for i, response_split in enumerate(response_splits)
        ]

    @property
    def context(self) -> List[Utterance]:
        return self.utterances[:-1]

    @property
    def response(self) -> Utterance:
        return self.utterances[-1]
    
    @property
    def character_count(self) -> int:
        return sum([len(utterance.text) for utterance in self.utterances])

    @abstractmethod
    def get_utterances(self, inp: str) -> List[Utterance]:
        pass

class DialogCC(Dialog):
    def get_utterances(self, inp: str) -> List[Utterance]:
        utterances = []
        ls = ast.literal_eval(inp)
        for dictionary in ls:
            utterance = Utterance(
                text=dictionary['utterance'], speaker=dictionary['speaker']
            )
            for image in dictionary['shared_image']:
                utterance.add_image(image['image_url'])
            utterances.append(utterance)
        return utterances
    
    def __repr__(self) -> str:
        return self.format_dialog()

class DialogData(Dataset):
    def __init__(
        self,
        path: str,
        to_filter: bool = False,
        to_replace: bool = False,
        image_path_by_url: Dict[str, str] = {},
        to_unroll: bool = False,
        min_images_per_dialog: int | None = None,
        n_samples: int | None = None,
        to_split: bool = False,
        ):
        self.to_unroll = to_unroll,
        self.to_split = to_split,
        self.to_filter = to_filter,
        self.to_replace = to_replace,
        self.n_samples = n_samples,
        self.min_images_per_dialog = min_images_per_dialog,
        self.path = path,
        self.dialogs = self.parse_raw_file()
        self.suffixes = []
        self.image_path_by_url = image_path_by_url
        self.dialogs = [self.preprocess(dialog) for dialog in self.dialogs]

        print("Number of dialogs before filtering: ", len(self.dialogs))
        if to_filter:
            assert (
                len(image_path_by_url) > 0
            ), 'image_path_by_url dictionary must be provided if to_filter is True'
            self.filter_and_replace_image_paths()
        
        print("Number of dialogs after filtering: ", len(self.dialogs))

        if to_unroll:
            unrolled_dialogs = []
            for dialog in tqdm(self.dialogs, desc='Unrolling dialogs'):
                unrolled_dialogs.extend(dialog.unroll())
            self.dialogs = unrolled_dialogs
        
        if min_images_per_dialog is not None:
            self.dialogs = [
                dialog
                for dialog in self.dialogs
                if sum([len(utterance.images) for utterance in dialog])
                >= min_images_per_dialog
            ]
        
        if n_samples is not None:
            self.dialogs = self.sample(n_samples)
        
        if to_split:
            split_dialogs = []
            suffixes = []
            for dialog in tqdm(self.dialogs, desc="Splitting dialogs"):
                for split_dialog, suffix in dialog.create_splits():
                    suffixes.append(suffix)
                    split_dialogs.append(split_dialog)
            self.dialogs = split_dialogs
            self.suffixes = suffixes
        else:
            self.suffixes = [None] * len(self.dialogs)
        self.dialog_suffix_by_id = {
            dialog.idx: (dialog, suffix) for dialog, suffix in zip(self.dialogs, self.suffixes)
        }
        print(
            f"Total dialogs: {len(self)}, Total suffixes: {len(self.suffixes)}, to_filter: {to_filter}, to_replace: {to_replace}, image_path_by_url (size): {len(image_path_by_url)}, to_unroll: {to_unroll}, min_images_per_dialog: {min_images_per_dialog}, n_samples: {n_samples}, to_split: {to_split}"
        )
        print(
            "HASH: ",
            hash_dataset(
                [
                    [utterance.text for utterance in dialog]
                    for dialog in self.dialogs
                ]
            ),
        )

    
    def __len__(self):
        return len(self.dialogs)
    
    def __getitem__(self, idx: int) -> Tuple[Dialog, str | None]:
        if self.to_split:
            return self.dialogs[idx], self.suffixes[idx]
        return self.dialogs[idx], None
    
    def __iter__(self):
        return iter(zip(self.dialogs, self.suffixes))
    
    def filter_and_replace_image_paths(self):
        """
        Filters out dialogs with images that are not in the image_path_by_url dictionary or whose image paths do not exist.
        """
        dialogs = []
        print("Reading image files")
        # print("Images dict", self.image_path_by_url[:10])
        for dialog in tqdm(self.dialogs, desc="Filtering dialogs"):
            not_found_flag = False
            utterances = []
            for utterance in dialog.utterances:
                # if ('http://www.lifewithdogs.tv/wp-content/uploads/2014/09/9.11.14-Worlds-Tallest-Dog-Dies5.jpg' in utterance.images):
                #     print("Type: ", type(self.image_path_by_url))
                #     print(utterance.images)
                # if ('http://www.lifewithdogs.tv/wp-content/uploads/2014/09/9.11.14-Worlds-Tallest-Dog-Dies5.jpg' in self.image_path_by_url):
                #     print('image present')
                #     print(self.image_path_by_url['http://www.lifewithdogs.tv/wp-content/uploads/2014/09/9.11.14-Worlds-Tallest-Dog-Dies5.jpg'])
                images_renamed = [
                    self.image_path_by_url[image]
                    for image in utterance.images
                    if image in self.image_path_by_url
                    and os.path.exists(self.image_path_by_url[image])
                ]
                # print(len(images_renamed), len(utterance.images))
                if len(images_renamed) != len(utterance.images):
                    not_found_flag = True
                utterances.append(Utterance(utterance.text, utterance.speaker))
                if self.to_replace:
                    utterances[-1].images = images_renamed
                else:
                    utterances[-1].images = utterance.images
            if not not_found_flag:
                dialogs.append(Dialog(utterances, dialog.idx))
        self.dialogs = dialogs


    def preprocess(self, dialog: Dialog) -> Dialog:
        """
        Preprocesses the dialog.
        """
        for utterance in dialog.utterances:
            utterance.text = utterance.text.lower()
        return dialog

    @abstractmethod
    def parse_raw_file(self) -> List[Dialog]:
        """
        Shall parse the dataset file and return a list of Dialog objects.
        """
        pass

    @abstractmethod
    def sample(self, n: int) -> List[Dialog]:
        """
        Shall sample n dialogs from each category of the dataset.
        """
        raise NotImplementedError("Sampling not yet implemented")

# class for fine-tuning on dialog data
class DialogCCData(DialogData):
    def id_prefix(self, path: str) -> str:
        return path.split('/')[-1].split('.')[0][:2]
    
    def parse_raw_file(self) -> List[Dialog]:
        path = self.path[0]
        df = pd.read_csv(path)
        dialogs = []
        for idx, row in tqdm(df.iterrows()):
            dialog = DialogCC(
                row['dialogue'],
                f"{self.id_prefix(path)}_{row['dialogue_id']}"
            )
            dialogs.append(dialog)
        return dialogs
    
    def sample(self, n: int) -> List[Dialog]:
        """
        Samples n dialogs from each category of the dataset.
        """
        dialogs_by_category = {}
        for dialog in self.dialogs:
            category = dialog.idx.split(":")[0]
            if category not in dialogs_by_category:
                dialogs_by_category[category] = []
            dialogs_by_category[category].append(dialog)
        print("Number of dialogs found in each category:")
        for category in dialogs_by_category:
            print(f"{category}: {len(dialogs_by_category[category])}")
        sampled_dialogs = []
        for category in dialogs_by_category:
            rng = np.random.default_rng(seed=42)
            sampled_dialogs += rng.choice(
                np.asarray(dialogs_by_category[category], dtype=object),
                n,
                replace=False,
            ).tolist()
        return sampled_dialogs

def sample(self, n: int) -> List[Dialog]:
        """
        Samples n dialogs from each category of the dataset.
        """
        dialogs_by_category = {}
        for dialog in self.dialogs:
            category = dialog.idx.split(":")[0]
            if category not in dialogs_by_category:
                dialogs_by_category[category] = []
            dialogs_by_category[category].append(dialog)
        print("Number of dialogs found in each category:")
        for category in dialogs_by_category:
            print(f"{category}: {len(dialogs_by_category[category])}")
        sampled_dialogs = []
        for category in dialogs_by_category:
            rng = np.random.default_rng(seed=42)
            sampled_dialogs += rng.choice(
                np.asarray(dialogs_by_category[category], dtype=object),
                n,
                replace=False,
            ).tolist()
        return sampled_dialogs


class DCCDataset(torch_dataset):
    def __init__(
        self,
        name,
        path,
        split='train',
        image_transform=None,
        tokenizer=None,
        num_image_tokens=0,
        user_key='human',
        assistant_key='gpt',
        load_image=True
    ):
        super().__init__()
        print("1", path)
        # self.path = osp.abspath(osp.expanduser(path))
        self.path = path
        print("2", self.path)
        self.split = split
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        
        self.user_key = user_key
        self.assistant_key = assistant_key
        
        self.load_image = load_image

        if self.tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-2.7b")

        if not IMAGE_TOKEN in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens(SPECIAL_TOKEN_DICT)

        self.setup_dataset()

    def setup_dataset(self):
        print("3", self.path)
        self.dialog_data = DialogCCData(
            path=os.path.join(self.path, 'train.csv' if self.split is 'train' else 'test.csv'),
            to_filter=True,
            to_replace=True,
            image_path_by_url=create_image_path_by_url(
                '/teamspace/studios/this_studio/DialogCC/DialogCC/image_names', '/teamspace/studios/this_studio/DialogCC/DialogCC/images'
            ),
            to_unroll=False,
            min_images_per_dialog=1,
            # n_samples=100,
            to_split=False
        )
        self.dialogs = self.dialog_data.dialogs
        self.suffixes = self.dialog_data.suffixes

    def transform_dialog_data_to_raw_data(
        self, dialog: Dialog, suffix: str | None
    ) -> List[Dict]:
        # if not self.test:
        #     assert suffix is not None, "suffix is None"
        # else:
        if suffix is None:
            suffix = "dummy"
        images = []
        for utterance in dialog.utterances:
            images.extend(utterance.images)
        # print("The utterances are: ")
        # for utterance in dialog.utterances:
        #     print(self.format_utterance(utterance))
        return {
            "system_prompt": "Complete the following conversation",
            "image": images,
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_utterance(utterance),
                }
                for utterance in dialog.utterances
            ]
            + [
                {
                    "from": "gpt",
                    "value": suffix,
                },
            ],
        }
    
    def format_utterance(self, utterance: Utterance) -> str:
        images_str = "<image>"*len(utterance.images)
        return f"{images_str}{utterance.text}"
    
    def image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def _read_image(self, images: List[str]):
        pixel_values = []
        for image_file in images:
            raw = Image.open(image_file)
            raw = raw.convert('RGB') if raw.mode != 'RGB' else raw

            if isinstance(self.image_transform, Compose):
                image = self.image_transform(raw)
            elif self.image_transform is not None:
                image = self.image_transform(raw, return_tensors='pt')
                image = image['pixel_values']
            pixel_values.append(image)
        return pixel_values

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        rng = np.random.RandomState(seed=42)
        # print("DIALOG: ", self.dialogs[index])
        conv = self.dialogs[index]
        unrolled = conv.unroll()
        unrolled_filter_min_one_image = list(
            filter(
                lambda x: sum([len(u.images) for u in x.utterances]) >= 1,
                unrolled,
            )
        )
        
        if len(unrolled_filter_min_one_image) == 0:
            print(f"This shouldnot happen, couldnot find a unroll of data that has at least one image \n DialogId: {conv.idx}")
            print(len(unrolled))
            # NOT IMPLEMENTED
            # print(self.transform_dialog_data_to_raw_data(conv, "dummy"))
            return self.__getitem__(rng.randint(len(self)))
        
    
        conv = unrolled_filter_min_one_image[rng.randint(len(unrolled_filter_min_one_image))]
        splits = conv.create_splits()
        random_split = splits[rng.randint(len(splits))]
        
        source = self.transform_dialog_data_to_raw_data(*random_split)
        images_ = []
        try:
            if "image" in source:
                # here we do not do any image preprocessing but rather
                # let the processor handle everything
                # in some cases this may cause slight differences
                # but should totally be fine (e.g., official llava-1.5 does padding,
                # but llava-1.5-hf (huggingface's implementation) does not)
                if isinstance(source["image"], list):
                    image_sources = source["image"]
                elif isinstance(source["image"], str):
                    image_sources = [source["image"]]
                else:
                    raise ValueError(f"Invalid image source type: {type(source['image'])}")

                for image_path in image_sources:
                    # if self.image_folder is not None:
                    #     image_path = os.path.join(
                    #         self.image_folder, image_path
                    #     )
                    if self.load_image:
                        image = Image.open(image_path).convert("RGB")
                        # sm745052: resize the image to 384x384
                        # print(f"Original image size: {image.size}")
                        image = image.resize((384, 384))
                        # print(f"Resized image size: {image.size}")
                        images_.append(image)
                    else:
                        images_.append(image_path)
        except Exception as e:
            print(f"Error in loading image/video: {e}")
            return self.__getitem__(rng.randint(len(self)))
        
        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        convs = []
        assert len(source["conversations"]) > 0, "No conversations found"
        assert source["conversations"][-1]["from"] == self.assistant_key, "Last utterance should be from assistant"
        assert set([conv["from"] for conv in source["conversations"][:-1]]) == set([self.user_key]), "All but last utterance should be from user"
        for i, conv in enumerate(source["conversations"]):
            convs.append(conv["value"])
        
        prefix_with_context = []
        images = []

        # dialog, suffix = self.dialogs[index], self.suffixes[index]
        dialog, suffix = random_split[0], random_split[1]

        for idx in range(len(dialog.utterances)):
            text = dialog.utterances[idx].text
            if dialog.utterances[idx].images != []:
                images.extend(dialog.utterances[idx].images)
                text = "".join(['<image>']*(2*len(dialog.utterances[idx].images))) + text
            prefix_with_context.append(text)      

        # check EOU token for OPT model
        prefix_with_context = '<s>'.join(prefix_with_context)
        temp = self.tokenizer(prefix_with_context, return_tensors='pt')
        l_pref = temp['input_ids'].shape[-1]

        # add the suffix to the prefix
        prefix_with_context = prefix_with_context + '<s>' + suffix
        inputs = self.tokenizer(prefix_with_context, return_tensors='pt')
        labels = inputs['input_ids'].unsqueeze(0).clone()
        labels[..., :l_pref] = -100 # suffix loss only
        # suffix_inputs = self.tokenizer(suffix, return_tensors='pt')
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        image_token_mask = (inputs['input_ids'] == image_token_id)

        return {
            'pixel_values': self._read_image(images),
            'dialog': prefix_with_context,
            'input_ids': inputs['input_ids'].unsqueeze(0),
            'attention_mask': inputs['attention_mask'].unsqueeze(0),
            'image_token_mask': image_token_mask.long(),
            'dialog_id': dialog.idx,
            'suffix': suffix,
            'labels': labels
        }
  

    def _add_image_tokens(self, inputs):
        if self.num_image_tokens > 0:
            tokens = [self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)] * self.num_image_tokens
            inputs['input_ids'] = tokens + inputs['input_ids']
            inputs['attention_mask'] = [1] * self.num_image_tokens + inputs['attention_mask']
        return inputs



def create_image_path_by_url(
    image_names_dir: str, images_dir: str
) -> Dict[str, str]:
    image_path_by_url = {}
    for file in os.listdir(image_names_dir):
        with open(os.path.join(image_names_dir, file)) as f:
            image_names = pd.read_csv(f, sep="\t", header=None, engine='python')
            for i in range(len(image_names)):
                image_path_by_url[image_names.iloc[i, 0]] = os.path.join(
                    images_dir, str(image_names.iloc[i, 1])
                )
    return image_path_by_url


class PyTorchWrapper(torch_dataset):  # Use Dataset[Any] to clarify for the type checker
    def __init__(self, hf_dataset: Dataset):
        self.hf_dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        return self.hf_dataset[idx]


class DialogDataModule(LightningDataModule):
    def __init__(self, config=dict()):
        super().__init__()
        self.config = config
        self.init_tokenizer()
        self.init_image_transform()
        self.load_splits()

    @property
    def loader_config(self):
        default_config = {
            'num_workers': 0,
            'pin_memory': False,
            'batch_size': 16,
        }
        return self.config.get('loader', default_config)

    @property
    def dataset_config(self):
        return self.config.get('dataset', dict())

    @property
    def model_config(self):
        return self.config.get('model', dict())

    def init_tokenizer(self):
        arch = self.model_config.get('text_encoder', 'facebook/opt-2.7b')
        self.tokenizer = GPT2Tokenizer.from_pretrained(arch)

    def init_image_transform(self):
        arch = self.model_config.get('image_encoder', 'microsoft/resnet-50')
        self.image_transform = get_image_transform(arch)

    def load_splits(self):
        self.train_data = self.load_split('train')
        self.val_data = self.load_split('val')

    def load_split(self, split):
        dataset_class = DCCDataset
        print("*******split********", split)
        return dataset_class(
            split=split,
            tokenizer=self.tokenizer,
            image_transform=self.image_transform,
            **self.dataset_config,
        )

    def train_dataloader(self):
        torch_compat_dataset = PyTorchWrapper(self.train_data)
        print("TRAIN DATA SIZE: ", self.train_data.__len__())
        return DataLoader(
            torch_compat_dataset,
            collate_fn=dialog_collate_fn,
            shuffle=False,
            **self.loader_config,
        )

    def val_dataloader(self):
        torch_compat_dataset = PyTorchWrapper(self.val_data)
        return DataLoader(
            torch_compat_dataset,
            collate_fn=dialog_collate_fn,
            shuffle=False,
            **self.loader_config
        )

    def predict_dataloader(self):
        return self.val_dataloader()

def dialog_collate_fn(batch):
    batch_size = len(batch)
    longest_prefix = max(x['input_ids'].numel() for x in batch)

    # Leave pixel_values logic untouched
    pixel_values = []
    for x in batch:
        pixel_values.extend(x['pixel_values'])  # Use extend to flatten the list

    if pixel_values:
        pixel_values = torch.cat(pixel_values)
    else:
        pixel_values = torch.empty(0)

    # Helper function to initialize padded tensors
    def init_helper(value, dtype, batch_size, max_length):
        return torch.full((batch_size, max_length), value, dtype=dtype)

    # Initialize padded tensors for prefix inputs
    input_ids = init_helper(PAD_TOKEN_ID, torch.long, batch_size, longest_prefix)
    attention_mask = init_helper(0, torch.long, batch_size, longest_prefix)
    image_token_mask = init_helper(0, torch.long, batch_size, longest_prefix)
    labels = init_helper(0, torch.long, batch_size, longest_prefix)

    # Fill the padded tensors
    for i in range(batch_size):
        prefix_length = batch[i]['input_ids'].numel()

        # Fill prefix tensors
        input_ids[i, :prefix_length] = batch[i]['input_ids']
        attention_mask[i, :prefix_length] = batch[i]['attention_mask']
        image_token_mask[i, :prefix_length] = batch[i]['image_token_mask']
        labels[i, :prefix_length] = batch[i]['labels']


    return {
        'pixel_values': pixel_values,  # Unchanged handling
        'input_ids': input_ids,        # Padded prefix + suffix input IDs
        'attention_mask': attention_mask,
        'image_token_mask': image_token_mask,
        'labels': labels,
        'item_ids': [x['dialog_id'] for x in batch],     # List of dialog IDs
        'suffix': [x['suffix'] for x in batch]        # Raw suffix text (optional)
    }

def dialog_collate_fn_(batch):
    batch_size = len(batch)
    longest_combined = max(x['input_ids'].numel() + x['suffix_ids'].numel() for x in batch)

    # Leave pixel_values logic untouched
    pixel_values = []
    for x in batch:
        pixel_values.extend(x['pixel_values'])  # Flatten the list

    if pixel_values:
        pixel_values = torch.cat(pixel_values)
    else:
        pixel_values = torch.empty(0)

    # Helper function to initialize padded tensors
    def init_helper(value, dtype, batch_size, max_length):
        return torch.full((batch_size, max_length), value, dtype=dtype)

    # Initialize combined tensors
    combined_input_ids = init_helper(PAD_TOKEN_ID, torch.long, batch_size, longest_combined)
    combined_attention_mask = init_helper(0, torch.long, batch_size, longest_combined)
    loss_mask = init_helper(0, torch.bool, batch_size, longest_combined)

    # Fill the combined tensors
    for i in range(batch_size):
        prefix_length = batch[i]['input_ids'].numel()
        suffix_length = batch[i]['suffix_ids'].numel()
        total_length = prefix_length + suffix_length
        print("Sizes: ", prefix_length, suffix_length, batch[i]['input_ids'].shape, batch[i]['suffix_ids'].shape)
        # Concatenate prefix and suffix
        combined_input_ids[i, :total_length] = torch.cat([batch[i]['input_ids'], batch[i]['suffix_ids']], dim=2)
        combined_attention_mask[i, :total_length] = 1  # Mask valid positions

        # Mask suffix positions for loss calculation
        loss_mask[i, prefix_length:total_length] = 1

    return {
        'pixel_values': pixel_values,
        'input_ids': combined_input_ids,        # Combined prefix and suffix
        'attention_mask': combined_attention_mask,
        'loss_mask': loss_mask,                # Mask for loss calculation
        'item_ids': [x['dialog_id'] for x in batch],  # Dialog IDs
    }

# if __name__ == '__main__':
#     path = 'data/DialogCC/test.csv'
#     dialog_data = DialogCCData(
#         path,
#         to_filter = True,
#         to_replace = True,
#         image_path_by_url=create_image_path_by_url(
#             'image_names', 'images'
#         ),
#         to_unroll=True,
#         min_images_per_dialog=1,
#         n_samples=100,
#         to_split=True,
#     )


#     print(len(dialog_data))

#     print(
#         "avg number of images = ",
#         sum(
#             [
#                 sum([len(utterance.images) for utterance in dialog])
#                 for dialog, _ in dialog_data
#             ]
#         )
#         / len(dialog_data)
#     )
#     print(
#         "sum of images = ",
#         sum(
#             [
#                 sum([len(utterance.images) for utterance in dialog])
#                 for dialog, _ in dialog_data
#             ]
#         ),
#     )
#     # avg number of characters per dialog
#     print(
#         "avg number of characters = ",
#         sum([dialog.character_count for dialog, _ in dialog_data])
#         / len(dialog_data)
#     )

#     print(dialog_data[0][0].format_dialog())
#     print(f"\nSuffix: {dialog_data[0][1]}")

    


