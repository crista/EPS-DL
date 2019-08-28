from keras import backend as K
import numpy as np

INPUT_SIZE = 10
OUTPUT_SIZE = 5
MAX_WORD_SIZE = 8
INPUT_VOCAB_SIZE = 12
BATCH_SIZE = 2
OUTPUT_SIZE = 5
MAX_WORD_SIZE = 4

def SpaceDetector2(x):
    print("x-sh", x.shape)
#    print("input: ", K.eval(x))

    sp_idx = 0#ctable.char_indices[' ']
    sp = np.zeros((INPUT_VOCAB_SIZE))
    sp[sp_idx] = 1

    filtered = x * sp
#    print("filtered:", K.eval(filtered))
    sp_positions = K.tf.where(K.tf.equal(filtered, 1)) # row indices
    print(sp_positions.shape)
#    print("sp-p:", K.eval(sp_positions))

    starts = sp_positions[:-1] + [0, 1, 0]
    stops = sp_positions[1:] + [0, 0, INPUT_VOCAB_SIZE]
    sizes = stops - starts + [1, 0, 0]
    where = K.tf.equal(sizes[:, 0], 1)
    starts = K.tf.boolean_mask(starts, where) # Remove multi-sample rows
    sizes = K.tf.boolean_mask(sizes, where) # Same
    where = K.tf.greater(sizes[:, 1], 0)
    starts = K.tf.boolean_mask(starts, where) # Remove words with 0 length (consecutive spaces)
    sizes = K.tf.boolean_mask(sizes, where) # Same

    print("starts:", starts, "sh:", starts.shape)
    print("stops:", stops)
    print("sizes:", sizes, "sh:", sizes.shape)

    slices = K.map_fn(lambda info: K.tf.pad(K.squeeze(K.slice(x, info[0], info[1]), 0), [[0, MAX_WORD_SIZE - info[1][1]], [0,0]], "CONSTANT"), [starts, sizes], dtype=float)
    return slices

def SpaceDetector(x):
    sp_idx = 0
    sp = np.zeros((INPUT_VOCAB_SIZE))
    sp[sp_idx] = 1

    def c2w(line):
        filtered = x * sp
        sp_positions = K.tf.where(K.tf.equal(filtered, 1)) # row indices

        starts = K.eval(sp_positions[:-1]) + [1, 0] 
        stops = K.eval(sp_positions[1:]) + [0, INPUT_VOCAB_SIZE] #[0, INPUT_VOCAB_SIZE])
        sizes = stops - starts

        slices = [K.slice(x, i, j) for i, j in zip(starts, sizes)]
        slices_padded = [K.tf.pad(s, [[0, MAX_WORD_SIZE - size], [0,0]], "CONSTANT") for s, size in zip(slices, sizes[:,0])]
        for _ in range(OUTPUT_SIZE-len(slices_padded)):
            slices_padded.append(K.variable(np.zeros((MAX_WORD_SIZE, INPUT_VOCAB_SIZE))))
        words = K.variable(slices_padded)
        return words
    return K.map_fn(c2w, x)

x = np.zeros((BATCH_SIZE, INPUT_SIZE, INPUT_VOCAB_SIZE))
x[0,0,0]=1 # space
x[0,1,1]=1 
x[0,2,3]=1
x[0,3,0]=1 # space
x[0,4,7]=1
x[0,5,9]=1
x[0,6,0]=1 # space
x[0,7,2]=1
x[0,8,4]=1
x[0,9,0]=1 # space

x[1,0,0]=1 # space
x[1,1,1]=1 
x[1,2,0]=1  # space
x[1,3,3]=1
x[1,4,7]=1
x[1,5,9]=1
x[1,6,0]=1 # space
x[1,7,2]=1
x[1,8,0]=1 # space
x[1,9,0]=1 # space

y=SpaceDetector2(K.variable(x))
print(K.eval(y))