#!/usr/bin/python3

# 🎩 tip: https://stackoverflow.com/a/8897648/1429450
# cf.: https://gregobase.selapa.net/?page_id=18#comment-66831
#   and "Top 49 chants most similar to the Requiem's gradual, sorted by cosine TF-IDF similarity of the GABC files.": https://forum.musicasacra.com/forum/discussion/comment/246225#Comment_246225

# NEED TO RUN load.py FIRST

import sys
if len(sys.argv) != 5:
    print("4 args required: similarity matrix shared memory ID, # matrix rows/cols, GregoBase ID, number of results")
    sys.exit(1)

import os, re
import numpy as np
from multiprocessing import shared_memory, resource_tracker

os.chdir('GABCs')
text_files = os.listdir()
documents = [open(f).read() for f in text_files]
arr_size = int(sys.argv[2])
num_elems = arr_size*(arr_size+1)//2  # n(n+1)/2 elements in lower-triangle (incl. diagonal)

try:
    existing_shm = shared_memory.SharedMemory(name=sys.argv[1])
except:
    print('<font color="red">Unable to open similarity matrix from shared memory.</font>')
    sys.exit(1)
resource_tracker.unregister(existing_shm._name, 'shared_memory') # Keep SHM persistent. https://stackoverflow.com/a/64916180/1429450
pairwise_similarity = np.ndarray((num_elems,), dtype=np.float64, buffer=existing_shm.buf)  # The lower-triangular matrix is in 1-D array representation.

# 🎩-tip for the following 2 functions: ChatGPT 4o https://chat.openai.com/share/75de2f76-cd12-4d4f-9e4a-154eac227407
def get_index(i, j, n):
    if i >= j:
        return i * (i + 1) // 2 + j
    else:
        return j * (j + 1) // 2 + i
def get_row_from_1d_array(lower_tri_elements, row_index, n):
    row = np.zeros(n)
    for j in range(n):
        index = get_index(row_index, j, n)
        row[j] = lower_tri_elements[index]
    return row

#top n similar files for document
filename=f'{sys.argv[3]}.gabc'
n = int(sys.argv[4])
idx = text_files.index(filename)

row = get_row_from_1d_array(pairwise_similarity,idx,arr_size)
topn = np.argsort(row)[::-1][:n+1]
for i in topn:
    gabc = documents[i]
    try:
        name = re.findall(r"name:([^;]*)", gabc)[0]
    except:
        pass
    gabc_id = text_files[i]
    gabc_id = int(gabc_id.removesuffix(".gabc"))
    print(f'<li><details><summary><a href="?id={gabc_id}">🔍</a> {round(row[i]*100)!s}% {name!s}</summary><a href="https://gregobase.selapa.net/chant.php?id={gabc_id}" target="_blank"><img src="https://gregobase.selapa.net/chant_img.php?id={gabc_id}" alt="{name!s}" loading="lazy"/></a></details></li>')
