Google Colab
================
----------

## Run tensorboard to visualize embeddings

- Taken from: [how-to-use-tensorboard-embedding-projector](https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector)

```{code-block} python
---
lineno-start: 1
emphasize-lines: 13, 14
---
import numpy as np
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

vectors = np.array([[0,0,1], [0,1,0], [1,0,0], [1,1,1]])
metadata = ['001', '010', '100', '111']  # labels
writer = SummaryWriter()
writer.add_embedding(vectors, metadata)
writer.close()

%load_ext tensorboard
%tensorboard --logdir=runs
```
