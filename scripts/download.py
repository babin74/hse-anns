import argparse, time, sys
from urllib.request import urlretrieve

resources = {
    'sift-128': (
        'https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift-128-euclidean.hdf5?download=true',
        'sift-128-euclidean.hdf5'
    ),
    'siftsmall-128': {
        'https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/siftsmall-128-euclidean.hdf5?download=true',
        'siftsmall-128-euclidean.hdf5'
    }
}

def run(args):
    dataset = args.dataset
    verbose = args.verbose
    
    if dataset not in resources:
        print(f"Unknown resource {dataset}")
        return
    
    url, name = resources[dataset]
    if verbose:
        print(f"Downloading {dataset}...")
        print(f"URL: {url}\nDIR: {name}")

    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()
    urlretrieve(url, f'./datasets/{name}', reporthook)
    print("Done!")
    

def add_subparser(subparsers):
    parser: argparse.ArgumentParser =\
        subparsers.add_parser('download', aliases=['dn', 'get'], help='download dataset')
    parser.add_argument('dataset', type=str, help='name of dataset to download')
    parser.set_defaults(func=run)