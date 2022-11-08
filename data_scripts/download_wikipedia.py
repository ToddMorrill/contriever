"""This script is a temporary solution to get up and running with Wikipedia data
before we resolve the issue below and start working with the December 20th, 2018
Wikipedia dump.
https://github.com/facebookresearch/contriever/issues/12

See Hugging Face for more details on this dataset.
https://huggingface.co/datasets/wikipedia

Examples:
    $ python download_wikipedia.py
"""
from datasets import load_dataset
import tqdm

def append_to_file(fh, content):
    fh.write(f'{content}\n')

def main():
    # this should take about 3 minutes to download on connection that gets 100mb/s
    wiki_data = load_dataset("wikipedia", "20220301.en")
    fh = open('en_XX.txt', 'w')
    # this will take about 10 minutes
    for idx, page in enumerate(tqdm.tqdm(wiki_data['train'])):
        append_to_file(fh, page['title'])
        append_to_file(fh, page['text'])
    fh.close()

if __name__ == '__main__':
    main()