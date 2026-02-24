import gzip
import json
import pickle
import random
import requests
import torch
from torch.utils.data import Dataset


GENRE_URL_DICT = {
    'poetry':                 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children':               'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic':         'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal':     'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography':      'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance':                'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult':            'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}


def load_reviews(url: str, head: int = 10000, sample_size: int = 2000) -> list:
    reviews = []
    count = 0
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with gzip.open(response.raw, 'rt', encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def download_dataset(cache_path: str = 'genre_reviews_dict.pickle',
                     head: int = 10000, sample_size: int = 2000) -> dict:
    try:
        genre_reviews_dict = pickle.load(open(cache_path, 'rb'))
        return genre_reviews_dict
    except FileNotFoundError:
        pass

    genre_reviews_dict = {}
    for genre, url in GENRE_URL_DICT.items():
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)

    pickle.dump(genre_reviews_dict, open(cache_path, 'wb'))
    return genre_reviews_dict


def split_data(genre_reviews_dict: dict,
               train_per_genre: int = 800,
               test_per_genre: int = 200,
               seed: int = 42) -> tuple:
    random.seed(seed)
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews_dict.items():
        reviews = random.sample(reviews, min(train_per_genre + test_per_genre, len(reviews)))
        for review in reviews[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in reviews[train_per_genre:train_per_genre + test_per_genre]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


class GoodreadsDataset(Dataset):

    def __init__(self, encodings, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)