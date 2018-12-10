import os
import sys
import zipfile
from six.moves import urllib
import requests

import pickle
import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def gpu_helper(gpu):
    if gpu >= -1:
        def to_gpu(x):
            x = x.cuda()
            return x
        return to_gpu
    else:
        def no_op(x):
            return x
        return no_op

def save_weights(module,file):
    weights = []
    for p in module.parameters():
        weights += [p.cpu().data.numpy()]
    pickle.dump(weights,open(file,'wb'))

def load_weights(module,file):
    weights = pickle.load(open(file,'rb'))
    for p,w in zip(module.parameters(),weights):
        p.data = torch.from_numpy(w)
    return weights

def detach_all(a):
    detached = []
    for ai in a:
        if isinstance(ai, list) or isinstance(ai, tuple):
            detached += [detach_all(ai)]
        else:
            detached += [ai.detach()]
    return detached

def simple_plot(data_1d, xlabel, ylabel, title, filepath):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(data_1d)),np.array(data_1d))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    fig.savefig(filepath)
    plt.close(fig)

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
# https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
