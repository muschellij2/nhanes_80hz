import numpy as np
from numpy import typing as npt
from scipy import signal
import scipy 

# BPF. There are extraneous coefficients as to match constants
# in ActiLife.
# Input data coefficients.
INPUT_COEFFICIENTS: npt.NDArray[np.float_] = np.array(
  [
    [
      -0.009341062898525,
      -0.025470289659360,
      -0.004235264826105,
      0.044152415456420,
      0.036493718347760,
      -0.011893961934740,
      -0.022917390623150,
      -0.006788163862310,
      0.000000000000000,
    ]
  ]
)

# Output data coefficients.
OUTPUT_COEFFICIENTS: npt.NDArray[np.float_] = np.array(
  [
    [
      1.00000000000000000000,
      -3.63367395910957000000,
      5.03689812757486000000,
      -3.09612247819666000000,
      0.50620507633883000000,
      0.32421701566682000000,
      -0.15685485875559000000,
      0.01949130205890000000,
      0.00000000000000000000,
    ]
  ]
)
x = INPUT_COEFFICIENTS[0, :]
y = OUTPUT_COEFFICIENTS[0, :]
print("making filter", flush = True)
zi = scipy.signal.lfilter_zi(x, y)
print("filter made", flush = True)
zi = zi.reshape((1, -1))

y
# zi = signal.lfilter_zi(INPUT_COEFFICIENTS[0, :], OUTPUT_COEFFICIENTS[0, :]).reshape(
#   (1, -1)
# )

