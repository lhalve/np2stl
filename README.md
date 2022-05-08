# np2stl
Create 3D STL files from numpy data

## Installation:
1. Clone the repository
```bash
git clone git@github.com:lhalve/np2stl.git
```
2. Add this repository to your PYTHONPATH environment variable
```bash
cd np2stl
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

## Usage:
Example usage for a numpy 2-dimensional histogram:
```python
import numpy as np
from np2stl import stl_builder

# create random numbers according to a 2d Gaussian distribution
rand = np.random.multivariate_normal([4, 7], np.diag([2, 3]), size=1000000)
# create a 2-dimensional histogram of these data
hist = np.histogram2d(rand[:, 0], rand[:, 1], bins=100)

# intialize the 3d object
builder = stl_builder.STLBuilder()
# feed the histogram to the 3d object
builder.from_numpy_2dhist(*hist)
# save the 3d object to a stl
builder.save("gauss_2d.stl")
```

## Output:
You can view .stl files with any 3d viewer. The example above creates an stl file similar to this:
