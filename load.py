#!/usr/bin/python3

# 🎩 tip: https://stackoverflow.com/a/8897648/1429450
# cf.: https://gregobase.selapa.net/?page_id=18#comment-66831
#   and 'Top 49 chants most similar to the Requiem's gradual, sorted by cosine TF-IDF similarity of the GABC files.': https://forum.musicasacra.com/forum/discussion/comment/246225#Comment_246225

from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import numpy as np
from pathlib import Path

npz_basename = 'lower_triangular'
npz_path = Path(f'{npz_basename}.npz')

def generateAndSaveSimilarityMatrix():
    text_files = [f for f in Path('./GABCs').glob('*.gabc')]

    documents = [f.read_text() for f in text_files]

    vectorizer = TfidfVectorizer()  # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
    tfidf = vectorizer.fit_transform(documents)

    pairwise_similarity = tfidf * tfidf.T

    # Extract the elements of the lower triangular part including the diagonal
    lower_triangular = pairwise_similarity.toarray()[np.tril_indices(pairwise_similarity.shape[0])]
    # Save it.
    np.savez(npz_path, lower_triangular)

def loadNpyFilesIntoSHM():
    from multiprocessing import shared_memory  # https://docs.python.org/3.12/library/multiprocessing.shared_memory.html#multiprocessing.shared_memory.SharedMemory.size

    npzfile = np.load(npz_path)
    loaded_matrix = npzfile['arr_0.npy']
    shm = shared_memory.SharedMemory(create=True, size=loaded_matrix.nbytes)

    # Copy the data into the shared memory block
    shared_array = np.ndarray(loaded_matrix.shape, dtype=loaded_matrix.dtype, buffer=shm.buf)
    np.copyto(shared_array, loaded_matrix)

    print(f'shm name: {shm.name}')
    print(f'matrix.shape: {loaded_matrix.shape}')
    Path('shm.name.txt').write_text(shm.name)

    print('Hit enter to exit and cleanup shared memory.')
    sys.stdin.readline()

    #cleanup
    shm.close()
    shm.unlink()

if not npz_path.exists():
    print(f'Generating similarity matrix and saving it as {npz_path}…')
    generateAndSaveSimilarityMatrix()

print(f'Loading {npz_path} into shared memory (shm)…')
loadNpyFilesIntoSHM()
