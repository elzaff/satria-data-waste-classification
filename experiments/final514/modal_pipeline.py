
"""Standalone SigLIP2 NaFlex Audited514 ErrorLogLoss training pipeline."""

from __future__ import annotations
import hashlib
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd


SEED = 2026
NAFLEX_ID = "google/siglip2-so400m-patch16-naflex"
NAFLEX_REVISION = "cc24074f717b612951c2dead130904ab9b65a81e"
_SWAPS = ({'R_387.jpg': 2,
                          'R_393.jpg': 2,
                          'R_533.jpg': 2,
                          'R_561.jpg': 2,
                          'R_7132.jpg': 2},
 {'O_10083.jpg': 0,
                           'O_10301.jpg': 0,
                           'O_10704.jpg': 0,
                           'O_11191.jpg': 0,
                           'O_11423.jpg': 0,
                           'O_1275.jpg': 0,
                           'O_1704.jpg': 0,
                           'O_1730.jpg': 0,
                           'O_1892.jpg': 0,
                           'O_1940.jpg': 0,
                           'O_1959.jpg': 0,
                           'O_1978.jpg': 0,
                           'O_3117.jpg': 0,
                           'O_322.jpg': 0,
                           'O_345.jpg': 0,
                           'O_396.jpg': 0,
                           'O_457.jpg': 0,
                           'O_476.jpg': 0,
                           'O_507.jpg': 0,
                           'O_516.jpg': 0,
                           'O_5347.jpg': 0,
                           'O_5700.jpg': 0,
                           'O_5890.jpg': 0,
                           'O_7293.jpg': 0,
                           'O_7935.jpg': 0,
                           'O_8577.jpg': 0,
                           'O_8580.jpg': 0,
                           'O_8640.jpg': 0,
                           'O_8643.jpg': 0,
                           'O_8752.jpg': 0,
                           'O_8770.jpg': 0,
                           'O_8791.jpg': 0,
                           'O_8826.jpg': 0,
                           'O_8846.jpg': 0,
                           'O_8880.jpg': 0,
                           'O_8919.jpg': 0,
                           'O_8922.jpg': 0,
                           'O_8972.jpg': 0,
                           'O_9031.jpg': 0,
                           'O_9083.jpg': 0,
                           'O_9132.jpg': 0,
                           'O_9137.jpg': 0,
                           'O_9170.jpg': 0,
                           'O_9175.jpg': 0,
                           'O_9195.jpg': 0,
                           'R_2451.jpg': 2,
                           'R_2478.jpg': 2,
                           'R_2585.jpg': 2,
                           'R_4602.jpg': 2,
                           'R_4708.jpg': 2,
                           'R_4730.jpg': 2,
                           'R_5044.jpg': 2,
                           'R_5087.jpg': 2,
                           'R_5095.jpg': 2,
                           'R_5248.jpg': 2,
                           'R_5299.jpg': 2,
                           'R_5319.jpg': 2,
                           'R_5340.jpg': 2,
                           'R_5414.jpg': 2,
                           'R_548.jpg': 2,
                           'R_5582.jpg': 2,
                           'R_5631.jpg': 2,
                           'R_566.jpg': 2,
                           'R_5682.jpg': 2,
                           'R_5704.jpg': 2,
                           'R_6041.jpg': 2,
                           'R_725.jpg': 2,
                           'R_8220.jpg': 2,
                           'R_8326.jpg': 2,
                           'R_8417.jpg': 2,
                           'R_9775.jpg': 2},
 {'O_8938.jpg': 0,
                           'O_8952.jpg': 0,
                           'O_8963.jpg': 0,
                           'O_8965.jpg': 0,
                           'O_8971.jpg': 0,
                           'O_8975.jpg': 0,
                           'O_8976.jpg': 0,
                           'O_8981.jpg': 0,
                           'O_8984.jpg': 0,
                           'O_8988.jpg': 0,
                           'O_8991.jpg': 0,
                           'O_8992.jpg': 0,
                           'O_8996.jpg': 0,
                           'O_8998.jpg': 0,
                           'O_9000.jpg': 0,
                           'O_9004.jpg': 0,
                           'O_9005.jpg': 0,
                           'O_9006.jpg': 0,
                           'O_9013.jpg': 0,
                           'O_9014.jpg': 0,
                           'O_9016.jpg': 0,
                           'O_9022.jpg': 0,
                           'O_9023.jpg': 0,
                           'O_9025.jpg': 0,
                           'O_9028.jpg': 0,
                           'O_9029.jpg': 0,
                           'O_9034.jpg': 0,
                           'O_9035.jpg': 0,
                           'O_9037.jpg': 0,
                           'O_9041.jpg': 0,
                           'O_9042.jpg': 0,
                           'O_9047.jpg': 0,
                           'O_9049.jpg': 0,
                           'O_9050.jpg': 0,
                           'O_9055.jpg': 0,
                           'O_9056.jpg': 0,
                           'O_9057.jpg': 0,
                           'O_9058.jpg': 0,
                           'O_9059.jpg': 0,
                           'O_9065.jpg': 0,
                           'O_9067.jpg': 0,
                           'O_9071.jpg': 0,
                           'O_9074.jpg': 0,
                           'O_9076.jpg': 0,
                           'O_9081.jpg': 0,
                           'O_9085.jpg': 0,
                           'O_9086.jpg': 0,
                           'O_9092.jpg': 0,
                           'O_9099.jpg': 0,
                           'O_9101.jpg': 0,
                           'O_9102.jpg': 0,
                           'O_9106.jpg': 0,
                           'O_9107.jpg': 0,
                           'O_9110.jpg': 0,
                           'O_9114.jpg': 0,
                           'O_9118.jpg': 0,
                           'O_9119.jpg': 0,
                           'O_9123.jpg': 0,
                           'O_9125.jpg': 0,
                           'O_9126.jpg': 0,
                           'O_9129.jpg': 0,
                           'O_9135.jpg': 0,
                           'O_9147.jpg': 0,
                           'O_9150.jpg': 0,
                           'O_9154.jpg': 0,
                           'O_9159.jpg': 0,
                           'O_9163.jpg': 0,
                           'O_9168.jpg': 0,
                           'O_9178.jpg': 0,
                           'O_9184.jpg': 0,
                           'O_9187.jpg': 0,
                           'O_9188.jpg': 0,
                           'O_9191.jpg': 0,
                           'O_9193.jpg': 0,
                           'O_9200.jpg': 0,
                           'O_9202.jpg': 0,
                           'O_9203.jpg': 0,
                           'O_9206.jpg': 0,
                           'O_9208.jpg': 0,
                           'O_9213.jpg': 0,
                           'O_9214.jpg': 0,
                           'O_9221.jpg': 0,
                           'O_9222.jpg': 0,
                           'O_9223.jpg': 0,
                           'O_9224.jpg': 0,
                           'O_9225.jpg': 0,
                           'O_9230.jpg': 0,
                           'O_9232.jpg': 0,
                           'O_9233.jpg': 0,
                           'O_9234.jpg': 0,
                           'O_9235.jpg': 0,
                           'O_9236.jpg': 0,
                           'O_9237.jpg': 0,
                           'O_9238.jpg': 0,
                           'O_9243.jpg': 0},
 {'O_8929.jpg': 0,
                         'O_8930.jpg': 0,
                         'O_8935.jpg': 0,
                         'O_8947.jpg': 0,
                         'O_9039.jpg': 0,
                         'O_9133.jpg': 0,
                         'O_9204.jpg': 0},
 {'O_8560.jpg': 0,
                            'O_8563.jpg': 0,
                            'O_8564.jpg': 0,
                            'O_8565.jpg': 0,
                            'O_8566.jpg': 0,
                            'O_8567.jpg': 0,
                            'O_8569.jpg': 0,
                            'O_8570.jpg': 0,
                            'O_8571.jpg': 0,
                            'O_8573.jpg': 0,
                            'O_8575.jpg': 0,
                            'O_8577.jpg': 0,
                            'O_8578.jpg': 0,
                            'O_8580.jpg': 0,
                            'O_8582.jpg': 0,
                            'O_8584.jpg': 0,
                            'O_8585.jpg': 0,
                            'O_8587.jpg': 0,
                            'O_8588.jpg': 0,
                            'O_8589.jpg': 0,
                            'O_8590.jpg': 0,
                            'O_8591.jpg': 0,
                            'O_8593.jpg': 0,
                            'O_8595.jpg': 0,
                            'O_8598.jpg': 0,
                            'O_8601.jpg': 0,
                            'O_8602.jpg': 0,
                            'O_8603.jpg': 0,
                            'O_8605.jpg': 0,
                            'O_8606.jpg': 0,
                            'O_8607.jpg': 0,
                            'O_8608.jpg': 0,
                            'O_8609.jpg': 0,
                            'O_8610.jpg': 0,
                            'O_8612.jpg': 0,
                            'O_8614.jpg': 0,
                            'O_8615.jpg': 0,
                            'O_8616.jpg': 0,
                            'O_8617.jpg': 0,
                            'O_8618.jpg': 0,
                            'O_8619.jpg': 0,
                            'O_8620.jpg': 0,
                            'O_8625.jpg': 0,
                            'O_8626.jpg': 0,
                            'O_8627.jpg': 0,
                            'O_8628.jpg': 0,
                            'O_8629.jpg': 0,
                            'O_8630.jpg': 0,
                            'O_8633.jpg': 0,
                            'O_8634.jpg': 0,
                            'O_8639.jpg': 0,
                            'O_8640.jpg': 0,
                            'O_8641.jpg': 0,
                            'O_8643.jpg': 0,
                            'O_8646.jpg': 0,
                            'O_8647.jpg': 0,

                            'O_8649.jpg': 0,
                            'O_8650.jpg': 0,
                            'O_8651.jpg': 0,
                            'O_8652.jpg': 0,
                            'O_8655.jpg': 0,
                            'O_8656.jpg': 0,
                            'O_8658.jpg': 0,
                            'O_8659.jpg': 0,
                            'O_8661.jpg': 0,
                            'O_8662.jpg': 0,
                            'O_8663.jpg': 0,
                            'O_8664.jpg': 0,
                            'O_8665.jpg': 0,
                            'O_8667.jpg': 0,
                            'O_8669.jpg': 0,
                            'O_8670.jpg': 0,
                            'O_8671.jpg': 0,
                            'O_8673.jpg': 0,
                            'O_8674.jpg': 0,
                            'O_8675.jpg': 0,
                            'O_8676.jpg': 0,
                            'O_8677.jpg': 0,
                            'O_8678.jpg': 0,
                            'O_8679.jpg': 0,
                            'O_8681.jpg': 0,
                            'O_8684.jpg': 0,
                            'O_8685.jpg': 0,
                            'O_8686.jpg': 0,
                            'O_8691.jpg': 0,
                            'O_8692.jpg': 0,
                            'O_8693.jpg': 0,
                            'O_8695.jpg': 0,
                            'O_8696.jpg': 0,
                            'O_8697.jpg': 0,
                            'O_8699.jpg': 0,
                            'O_8700.jpg': 0,
                            'O_8701.jpg': 0,
                            'O_8702.jpg': 0,
                            'O_8703.jpg': 0,
                            'O_8705.jpg': 0,
                            'O_8706.jpg': 0,
                            'O_8707.jpg': 0,
                            'O_8708.jpg': 0,
                            'O_8709.jpg': 0,
                            'O_8712.jpg': 0,
                            'O_8714.jpg': 0,
                            'O_8715.jpg': 0,
                            'O_8717.jpg': 0,
                            'O_8718.jpg': 0,
                            'O_8719.jpg': 0,
                            'O_8721.jpg': 0,
                            'O_8724.jpg': 0,
                            'O_8727.jpg': 0,
                            'O_8728.jpg': 0,
                            'O_8729.jpg': 0,
                            'O_8730.jpg': 0,
                            'O_8731.jpg': 0,
                            'O_8732.jpg': 0,
                            'O_8734.jpg': 0,
                            'O_8735.jpg': 0,
                            'O_8740.jpg': 0,
                            'O_8741.jpg': 0,
                            'O_8742.jpg': 0,
                            'O_8743.jpg': 0,
                            'O_8746.jpg': 0,
                            'O_8747.jpg': 0,
                            'O_8749.jpg': 0,
                            'O_8750.jpg': 0,
                            'O_8751.jpg': 0,
                            'O_8752.jpg': 0,
                            'O_8753.jpg': 0,
                            'O_8755.jpg': 0,
                            'O_8756.jpg': 0,
                            'O_8758.jpg': 0,
                            'O_8759.jpg': 0,
                            'O_8760.jpg': 0,
                            'O_8761.jpg': 0,
                            'O_8763.jpg': 0,
                            'O_8764.jpg': 0,
                            'O_8765.jpg': 0,
                            'O_8766.jpg': 0,
                            'O_8767.jpg': 0,
                            'O_8768.jpg': 0,
                            'O_8770.jpg': 0,
                            'O_8772.jpg': 0,
                            'O_8774.jpg': 0,
                            'O_8775.jpg': 0,
                            'O_8776.jpg': 0,
                            'O_8777.jpg': 0,
                            'O_8779.jpg': 0,
                            'O_8780.jpg': 0,
                            'O_8782.jpg': 0,
                            'O_8783.jpg': 0,
                            'O_8784.jpg': 0,
                            'O_8785.jpg': 0,
                            'O_8786.jpg': 0,
                            'O_8787.jpg': 0,
                            'O_8789.jpg': 0,
                            'O_8790.jpg': 0,
                            'O_8791.jpg': 0,
                            'O_8792.jpg': 0,
                            'O_8793.jpg': 0,
                            'O_8794.jpg': 0,
                            'O_8795.jpg': 0,
                            'O_8796.jpg': 0,
                            'O_8797.jpg': 0,
                            'O_8798.jpg': 0,
                            'O_8799.jpg': 0,
                            'O_8800.jpg': 0,
                            'O_8801.jpg': 0,
                            'O_8802.jpg': 0,
                            'O_8803.jpg': 0,
                            'O_8804.jpg': 0,
                            'O_8806.jpg': 0,
                            'O_8807.jpg': 0,
                            'O_8808.jpg': 0,
                            'O_8809.jpg': 0,
                            'O_8810.jpg': 0,
                            'O_8812.jpg': 0,
                            'O_8814.jpg': 0,
                            'O_8815.jpg': 0,
                            'O_8817.jpg': 0,
                            'O_8818.jpg': 0,
                            'O_8819.jpg': 0,
                            'O_8820.jpg': 0,
                            'O_8821.jpg': 0,
                            'O_8824.jpg': 0,
                            'O_8825.jpg': 0,
                            'O_8826.jpg': 0,
                            'O_8827.jpg': 0,
                            'O_8828.jpg': 0,
                            'O_8830.jpg': 0,
                            'O_8831.jpg': 0,
                            'O_8832.jpg': 0,
                            'O_8833.jpg': 0,
                            'O_8834.jpg': 0,
                            'O_8836.jpg': 0,
                            'O_8839.jpg': 0,
                            'O_8840.jpg': 0,
                            'O_8841.jpg': 0,
                            'O_8842.jpg': 0,
                            'O_8843.jpg': 0,
                            'O_8845.jpg': 0,
                            'O_8846.jpg': 0,
                            'O_8847.jpg': 0,
                            'O_8848.jpg': 0,
                            'O_8849.jpg': 0,
                            'O_8852.jpg': 0,
                            'O_8853.jpg': 0,
                            'O_8855.jpg': 0,
                            'O_8856.jpg': 0,
                            'O_8857.jpg': 0,
                            'O_8859.jpg': 0,
                            'O_8861.jpg': 0,
                            'O_8862.jpg': 0,
                            'O_8863.jpg': 0,
                            'O_8864.jpg': 0,
                            'O_8865.jpg': 0,
                            'O_8866.jpg': 0,
                            'O_8867.jpg': 0,
                            'O_8868.jpg': 0,
                            'O_8869.jpg': 0,
                            'O_8872.jpg': 0,
                            'O_8873.jpg': 0,
                            'O_8874.jpg': 0,
                            'O_8875.jpg': 0,
                            'O_8876.jpg': 0,
                            'O_8877.jpg': 0,
                            'O_8878.jpg': 0,
                            'O_8879.jpg': 0,
                            'O_8880.jpg': 0,
                            'O_8882.jpg': 0,
                            'O_8883.jpg': 0,
                            'O_8891.jpg': 0,
                            'O_8895.jpg': 0,
                            'O_8896.jpg': 0,
                            'O_8897.jpg': 0,
                            'O_8900.jpg': 0,
                            'O_8901.jpg': 0,
                            'O_8902.jpg': 0,
                            'O_8904.jpg': 0,
                            'O_8906.jpg': 0,
                            'O_8907.jpg': 0,
                            'O_8908.jpg': 0,
                            'O_8909.jpg': 0,
                            'O_8910.jpg': 0,
                            'O_8911.jpg': 0,
                            'O_8912.jpg': 0,
                            'O_8913.jpg': 0,
                            'O_8914.jpg': 0,
                            'O_8915.jpg': 0,
                            'O_8916.jpg': 0,
                            'O_8917.jpg': 0,
                            'O_8918.jpg': 0,
                            'O_8919.jpg': 0,
                            'O_8920.jpg': 0,
                            'O_8921.jpg': 0,
                            'O_8922.jpg': 0,
                            'O_8923.jpg': 0,
                            'O_8924.jpg': 0,
                            'O_8925.jpg': 0,
                            'O_8926.jpg': 0,
                            'O_8927.jpg': 0},
 {'R_697.jpg': 2},
 {'R_1257.jpg': 2,
                        'R_2761.jpg': 2,
                        'R_302.jpg': 2,
                        'R_4528.jpg': 2,
                        'R_4583.jpg': 2,
                        'R_4638.jpg': 2,
                        'R_4869.jpg': 2,
                        'R_5569.jpg': 2,
                        'R_621.jpg': 2,
                        'R_678.jpg': 2,
                        'R_6935.jpg': 2,
                        'R_7136.jpg': 2,
                        'R_7170.jpg': 2,
                        'R_7226.jpg': 2,
                        'R_8257.jpg': 2,
                        'R_8327.jpg': 2,
                        'R_839.jpg': 2,
                        'R_8580.jpg': 2,
                        'R_9537.jpg': 2,
                        'R_9817.jpg': 2,
                        'R_9828.jpg': 2,
                        'R_9957.jpg': 2},
 {'O_152.jpg': 0,
                              'O_1678.jpg': 0,
                              'O_8574.jpg': 0,
                              'O_8636.jpg': 0,
                              'O_8653.jpg': 0,
                              'O_8805.jpg': 0},
 {'R_8295.jpg': 1},
 {'O_7776.jpg': 1})

FEATURE_SWAPS = dict(_SWAPS[4])
ESTABLISHED_SWAPS = {}
for _swaps in _SWAPS[:6]:
    ESTABLISHED_SWAPS.update(_swaps)
FOOD_CONTENT_SWAPS = dict(_SWAPS[6])
PROCESSED_FIBER_SWAPS = dict(_SWAPS[7])
ELECTRONIC_SWAPS = {**_SWAPS[8], **_SWAPS[9]}
FINAL_SWAPS = {
    **ESTABLISHED_SWAPS,
    **FOOD_CONTENT_SWAPS,
    **PROCESSED_FIBER_SWAPS,
    **ELECTRONIC_SWAPS,
}
del _SWAPS, _swaps


NEW_MATERIAL_SWAPS = {
    "O_6862.jpg": 0, "O_1909.jpg": 0, "O_9199.jpg": 0,
    "O_9201.jpg": 0, "O_9242.jpg": 0, "O_8899.jpg": 0,
}
BASELINE_FINAL_SWAPS = dict(FINAL_SWAPS)
FINAL_SWAPS.update(NEW_MATERIAL_SWAPS)
VALID462_SWAPS = dict(FINAL_SWAPS)

MATERIAL_EXTENSION_SWAPS = {
    **{f"O_{number}.jpg": 0 for number in (
        9108, 9212, 9032, 9020, 8966, 9093, 9158, 9080,
        9063, 9002, 9062, 8948, 8995, 9077, 9027, 9018,
        9179, 9130, 9151, 9015, 9180, 8961, 9149, 9177,
        6351, 9919, 4055, 6229, 1276, 7672, 5056, 1627,
        4764, 1564, 2734, 7709, 1652, 1864, 1725, 126,
    )},
    "O_7910.jpg": 1,
}
FINAL_SWAPS.update(MATERIAL_EXTENSION_SWAPS)
AUDITED503_SWAPS = dict(FINAL_SWAPS)

FOOD_ORGANIC_EXTENSION_SWAPS = {
    **{f"R_{number}.jpg": 2 for number in (
        2676, 2623, 6995, 631, 7205, 7225, 7156, 7141,
        386, 4914, 658,
    )},
}
FINAL_SWAPS.update(FOOD_ORGANIC_EXTENSION_SWAPS)

HEAD_CANDIDATES = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


AUDIT_FILENAMES = frozenset(['O_8560.jpg',
 'O_8561.jpg',
 'O_8562.jpg',
 'O_8563.jpg',
 'O_8564.jpg',
 'O_8565.jpg',
 'O_8566.jpg',
 'O_8567.jpg',
 'O_8568.jpg',
 'O_8569.jpg',
 'O_8570.jpg',
 'O_8571.jpg',
 'O_8572.jpg',
 'O_8573.jpg',
 'O_8574.jpg',
 'O_8575.jpg',
 'O_8576.jpg',
 'O_8577.jpg',
 'O_8578.jpg',
 'O_8579.jpg',
 'O_8580.jpg',
 'O_8581.jpg',
 'O_8582.jpg',
 'O_8583.jpg',
 'O_8584.jpg',
 'O_8585.jpg',
 'O_8586.jpg',
 'O_8587.jpg',
 'O_8588.jpg',
 'O_8589.jpg',
 'O_8590.jpg',
 'O_8591.jpg',
 'O_8592.jpg',
 'O_8593.jpg',
 'O_8594.jpg',
 'O_8595.jpg',
 'O_8596.jpg',
 'O_8597.jpg',
 'O_8598.jpg',
 'O_8599.jpg',
 'O_8600.jpg',
 'O_8601.jpg',
 'O_8602.jpg',
 'O_8603.jpg',
 'O_8604.jpg',
 'O_8605.jpg',
 'O_8606.jpg',
 'O_8607.jpg',
 'O_8608.jpg',
 'O_8609.jpg',
 'O_8610.jpg',
 'O_8611.jpg',
 'O_8612.jpg',
 'O_8613.jpg',
 'O_8614.jpg',
 'O_8615.jpg',
 'O_8616.jpg',
 'O_8617.jpg',
 'O_8618.jpg',
 'O_8619.jpg',
 'O_8620.jpg',
 'O_8621.jpg',
 'O_8622.jpg',
 'O_8623.jpg',
 'O_8624.jpg',
 'O_8625.jpg',
 'O_8626.jpg',
 'O_8627.jpg',
 'O_8628.jpg',
 'O_8629.jpg',
 'O_8630.jpg',
 'O_8631.jpg',
 'O_8632.jpg',
 'O_8633.jpg',
 'O_8634.jpg',
 'O_8635.jpg',
 'O_8636.jpg',
 'O_8637.jpg',
 'O_8638.jpg',
 'O_8639.jpg',
 'O_8640.jpg',
 'O_8641.jpg',
 'O_8642.jpg',
 'O_8643.jpg',
 'O_8644.jpg',
 'O_8645.jpg',
 'O_8646.jpg',
 'O_8647.jpg',
 'O_8648.jpg',
 'O_8649.jpg',
 'O_8650.jpg',
 'O_8651.jpg',
 'O_8652.jpg',
 'O_8653.jpg',
 'O_8654.jpg',
 'O_8655.jpg',
 'O_8656.jpg',
 'O_8657.jpg',
 'O_8658.jpg',
 'O_8659.jpg',
 'O_8660.jpg',
 'O_8661.jpg',
 'O_8662.jpg',
 'O_8663.jpg',
 'O_8664.jpg',
 'O_8665.jpg',
 'O_8666.jpg',
 'O_8667.jpg',
 'O_8668.jpg',
 'O_8669.jpg',
 'O_8670.jpg',
 'O_8671.jpg',
 'O_8672.jpg',
 'O_8673.jpg',
 'O_8674.jpg',
 'O_8675.jpg',
 'O_8676.jpg',
 'O_8677.jpg',
 'O_8678.jpg',
 'O_8679.jpg',
 'O_8680.jpg',
 'O_8681.jpg',
 'O_8682.jpg',
 'O_8683.jpg',
 'O_8684.jpg',
 'O_8685.jpg',
 'O_8686.jpg',
 'O_8687.jpg',
 'O_8688.jpg',
 'O_8689.jpg',
 'O_8690.jpg',
 'O_8691.jpg',
 'O_8692.jpg',
 'O_8693.jpg',
 'O_8694.jpg',
 'O_8695.jpg',
 'O_8696.jpg',
 'O_8697.jpg',
 'O_8698.jpg',
 'O_8699.jpg',
 'O_8700.jpg',
 'O_8701.jpg',
 'O_8702.jpg',
 'O_8703.jpg',
 'O_8704.jpg',
 'O_8705.jpg',
 'O_8706.jpg',
 'O_8707.jpg',
 'O_8708.jpg',
 'O_8709.jpg',
 'O_8710.jpg',
 'O_8711.jpg',
 'O_8712.jpg',
 'O_8713.jpg',
 'O_8714.jpg',
 'O_8715.jpg',
 'O_8716.jpg',
 'O_8717.jpg',
 'O_8718.jpg',
 'O_8719.jpg',
 'O_8720.jpg',
 'O_8721.jpg',
 'O_8722.jpg',
 'O_8723.jpg',
 'O_8724.jpg',
 'O_8725.jpg',
 'O_8726.jpg',
 'O_8727.jpg',
 'O_8728.jpg',
 'O_8729.jpg',
 'O_8730.jpg',
 'O_8731.jpg',
 'O_8732.jpg',
 'O_8733.jpg',
 'O_8734.jpg',
 'O_8735.jpg',
 'O_8736.jpg',
 'O_8737.jpg',
 'O_8738.jpg',
 'O_8739.jpg',
 'O_8740.jpg',
 'O_8741.jpg',
 'O_8742.jpg',
 'O_8743.jpg',
 'O_8744.jpg',
 'O_8745.jpg',
 'O_8746.jpg',
 'O_8747.jpg',
 'O_8748.jpg',
 'O_8749.jpg',
 'O_8750.jpg',
 'O_8751.jpg',
 'O_8752.jpg',
 'O_8753.jpg',
 'O_8754.jpg',
 'O_8755.jpg',
 'O_8756.jpg',
 'O_8757.jpg',
 'O_8758.jpg',
 'O_8759.jpg',
 'O_8760.jpg',
 'O_8761.jpg',
 'O_8762.jpg',
 'O_8763.jpg',
 'O_8764.jpg',
 'O_8765.jpg',
 'O_8766.jpg',
 'O_8767.jpg',
 'O_8768.jpg',
 'O_8769.jpg',
 'O_8770.jpg',
 'O_8771.jpg',
 'O_8772.jpg',
 'O_8773.jpg',
 'O_8774.jpg',
 'O_8775.jpg',
 'O_8776.jpg',
 'O_8777.jpg',
 'O_8778.jpg',
 'O_8779.jpg',
 'O_8780.jpg',
 'O_8781.jpg',
 'O_8782.jpg',
 'O_8783.jpg',
 'O_8784.jpg',
 'O_8785.jpg',
 'O_8786.jpg',
 'O_8787.jpg',
 'O_8788.jpg',
 'O_8789.jpg',
 'O_8790.jpg',
 'O_8791.jpg',
 'O_8792.jpg',
 'O_8793.jpg',
 'O_8794.jpg',
 'O_8795.jpg',
 'O_8796.jpg',
 'O_8797.jpg',
 'O_8798.jpg',
 'O_8799.jpg',
 'O_8800.jpg',

 'O_8801.jpg',
 'O_8802.jpg',
 'O_8803.jpg',
 'O_8804.jpg',
 'O_8805.jpg',
 'O_8806.jpg',
 'O_8807.jpg',
 'O_8808.jpg',
 'O_8809.jpg',
 'O_8810.jpg',
 'O_8811.jpg',
 'O_8812.jpg',
 'O_8813.jpg',
 'O_8814.jpg',
 'O_8815.jpg',
 'O_8816.jpg',
 'O_8817.jpg',
 'O_8818.jpg',
 'O_8819.jpg',
 'O_8820.jpg',
 'O_8821.jpg',
 'O_8822.jpg',
 'O_8823.jpg',
 'O_8824.jpg',
 'O_8825.jpg',
 'O_8826.jpg',
 'O_8827.jpg',
 'O_8828.jpg',
 'O_8829.jpg',
 'O_8830.jpg',
 'O_8831.jpg',
 'O_8832.jpg',
 'O_8833.jpg',
 'O_8834.jpg',
 'O_8835.jpg',
 'O_8836.jpg',
 'O_8837.jpg',
 'O_8838.jpg',
 'O_8839.jpg',
 'O_8840.jpg',
 'O_8841.jpg',
 'O_8842.jpg',
 'O_8843.jpg',
 'O_8844.jpg',
 'O_8845.jpg',
 'O_8846.jpg',
 'O_8847.jpg',
 'O_8848.jpg',
 'O_8849.jpg',
 'O_8850.jpg',
 'O_8851.jpg',
 'O_8852.jpg',
 'O_8853.jpg',
 'O_8854.jpg',
 'O_8855.jpg',
 'O_8856.jpg',
 'O_8857.jpg',
 'O_8858.jpg',
 'O_8859.jpg',
 'O_8860.jpg',
 'O_8861.jpg',
 'O_8862.jpg',
 'O_8863.jpg',
 'O_8864.jpg',
 'O_8865.jpg',
 'O_8866.jpg',
 'O_8867.jpg',
 'O_8868.jpg',
 'O_8869.jpg',
 'O_8870.jpg',
 'O_8871.jpg',
 'O_8872.jpg',
 'O_8873.jpg',
 'O_8874.jpg',
 'O_8875.jpg',
 'O_8876.jpg',
 'O_8877.jpg',
 'O_8878.jpg',
 'O_8879.jpg',
 'O_8880.jpg',
 'O_8881.jpg',
 'O_8882.jpg',
 'O_8883.jpg',
 'O_8884.jpg',
 'O_8885.jpg',
 'O_8886.jpg',
 'O_8887.jpg',
 'O_8888.jpg',
 'O_8889.jpg',
 'O_8890.jpg',
 'O_8891.jpg',
 'O_8892.jpg',
 'O_8893.jpg',
 'O_8894.jpg',
 'O_8895.jpg',
 'O_8896.jpg',
 'O_8897.jpg',
 'O_8898.jpg',
 'O_8899.jpg',
 'O_8900.jpg',
 'O_8901.jpg',
 'O_8902.jpg',
 'O_8903.jpg',
 'O_8904.jpg',
 'O_8905.jpg',
 'O_8906.jpg',
 'O_8907.jpg',
 'O_8908.jpg',
 'O_8909.jpg',
 'O_8910.jpg',
 'O_8911.jpg',
 'O_8912.jpg',
 'O_8913.jpg',
 'O_8914.jpg',
 'O_8915.jpg',
 'O_8916.jpg',
 'O_8917.jpg',
 'O_8918.jpg',
 'O_8919.jpg',
 'O_8920.jpg',
 'O_8921.jpg',
 'O_8922.jpg',
 'O_8923.jpg',
 'O_8924.jpg',
 'O_8925.jpg',
 'O_8926.jpg',
 'O_8927.jpg'])


def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def discover_dataset(root):
    root = Path(root)
    folders = {
        0: root / "train" / "0_Recyclable",
        1: root / "train" / "1_Electronic",
        2: root / "train" / "2_Organic",
    }
    if not all(folder.is_dir() for folder in folders.values()):
        raise FileNotFoundError(f"Raw BDC2026 train folders not found under {root}")
    rows = []
    for label, folder in folders.items():
        for path in sorted(item for item in folder.iterdir() if item.is_file()):
            rows.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "original_label": label,
                    "group": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = pd.DataFrame(rows)
    inference_paths = sorted(
        (item for item in (root / "test").iterdir() if item.is_file()),
        key=lambda path: int(path.stem),
    )
    assert len(manifest) == 26_527 and manifest.filename.is_unique
    assert manifest.original_label.value_counts().sort_index().to_dict() == {
        0: 9_999,
        1: 3_961,
        2: 12_567,
    }
    assert len(inference_paths) == 1_458
    assert [int(path.stem) for path in inference_paths] == list(range(1, 1_459))
    return manifest, inference_paths


def apply_swaps(manifest, swaps):
    index = {name: i for i, name in enumerate(manifest.filename)}
    labels = manifest.original_label.to_numpy(int).copy()
    for filename, label in swaps.items():
        if filename not in index:
            raise KeyError(f"Embedded swap target absent: {filename}")
        labels[index[filename]] = int(label)
    return labels


def build_manifests(manifest):
    feature_labels = apply_swaps(manifest, FEATURE_SWAPS)
    final_labels = apply_swaps(manifest, FINAL_SWAPS)
    original = manifest.original_label.to_numpy(int)
    assert int(np.sum(feature_labels != original)) == 259
    assert int(np.sum(final_labels != original)) == 514
    changed = np.flatnonzero(original != final_labels)
    audit = [
        {
            "filename": manifest.filename.iloc[row],
            "old_label": int(original[row]),
            "new_label": int(final_labels[row]),
        }
        for row in changed
    ]
    return feature_labels, final_labels, pd.DataFrame(audit)


def stable_folds(manifest):
    from sklearn.model_selection import StratifiedGroupKFold

    folds = np.full(len(manifest), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (_, valid) in enumerate(
        splitter.split(
            manifest.filename,
            manifest.original_label,
            groups=manifest.group,
        )
    ):
        folds[valid] = fold
    assert (folds == 0).sum() == 5_308 and np.all(folds >= 0)
    return folds


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)



def hierarchical_probability(class_logits, binary_logits, alpha=0.8):
    base = softmax(class_logits)
    q = 1.0 / (1.0 + np.exp(-binary_logits))
    hierarchical = np.column_stack(
        [(1 - base[:, 1]) * q, base[:, 1], (1 - base[:, 1]) * (1 - q)]
    )
    return alpha * base + (1 - alpha) * hierarchical


def metrics(labels, probability):
    from sklearn.metrics import f1_score

    prediction = probability.argmax(1)
    return {
        "errors": int(np.sum(prediction != labels)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
    }


def train_extract_naflex(
    manifest, inference_paths, feature_labels, folds, device, work_dir,
    checkpoint_commit=None,
):
    from datetime import datetime, timezone

    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import ColorJitter
    from transformers import (
        AutoImageProcessor,
        Siglip2VisionModel,
        get_cosine_schedule_with_warmup,
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    processor = AutoImageProcessor.from_pretrained(
        NAFLEX_ID, revision=NAFLEX_REVISION, use_fast=True
    )
    processor.save_pretrained(work_dir)

    class Images(Dataset):
        def __init__(self, paths, labels=None, training=False, weights=None):
            self.paths = list(paths)
            self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
            self.training = training
            self.weights = np.ones(len(self.paths), np.float32) if weights is None else np.asarray(weights, np.float32)
            self.jitter = ColorJitter(0.10, 0.10, 0.10, 0.02)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source:
                image = source.convert("RGB")
            if self.training:
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
                image = self.jitter(image)
            label = -1 if self.labels is None else int(self.labels[index])
            return image, label, float(self.weights[index])

    def collate(rows):
        images, labels, weights = zip(*rows)
        batch = processor(
            images=list(images), return_tensors="pt", max_num_patches=256
        )
        return batch, torch.tensor(labels), torch.tensor(weights)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Siglip2VisionModel.from_pretrained(
                NAFLEX_ID,
                revision=NAFLEX_REVISION,
                attn_implementation="sdpa",
            )
            dimension = self.backbone.config.hidden_size
            self.classifier = nn.Linear(dimension, 3)
            self.binary = nn.Linear(dimension, 1)
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        def forward(self, batch):
            output = self.backbone(
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch["pixel_attention_mask"],
                spatial_shapes=batch["spatial_shapes"],
            )
            features = output.pooler_output
            return self.classifier(features), self.binary(features).squeeze(1), features

    def loader(paths, labels=None, training=False, weights=None, epoch_seed=SEED):
        generator = torch.Generator().manual_seed(epoch_seed)

        def worker_seed(worker_id):
            value = epoch_seed + worker_id
            random.seed(value)
            np.random.seed(value)
            torch.manual_seed(value)

        return DataLoader(
            Images(paths, labels, training, weights),
            batch_size=32,
            shuffle=training,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate,
            worker_init_fn=worker_seed,
            generator=generator,
        )

    core_model = Model().to(device)
    model = core_model
    train_mask, valid_mask = folds != 0, folds == 0
    train_paths = manifest.path.to_numpy()[train_mask]
    valid_paths = manifest.path.to_numpy()[valid_mask]
    train_labels = feature_labels[train_mask]
    valid_labels = feature_labels[valid_mask]
    counts = np.bincount(train_labels, minlength=3)
    values = np.sqrt(counts.sum() / np.maximum(counts, 1))
    class_weights = torch.tensor(
        values / values.mean(), dtype=torch.float32, device=device
    )
    validation_loader = loader(valid_paths, valid_labels)

    def predict(data_loader, return_features=False):
        model.eval()
        logits, binaries, features = [], [], []
        with torch.inference_mode():
            for batch, _, _ in data_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    class_logits, binary_logits, encoded = model(batch)
                logits.append(class_logits.float().cpu().numpy())
                binaries.append(binary_logits.float().cpu().numpy())
                if return_features:
                    features.append(encoded.float().cpu().numpy())
        output = (np.concatenate(logits), np.concatenate(binaries))
        return (*output, np.concatenate(features)) if return_features else output

    best_score = -1.0
    checkpoint = Path(work_dir) / "naflex_best.pt"
    history = []
    global_epoch = 0
    for phase, epochs in (("head", 2), ("partial", 4)):
        if phase == "partial":
            for block in core_model.backbone.vision_model.encoder.layers[-4:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for layer in (core_model.backbone.vision_model.post_layernorm, core_model.backbone.vision_model.head):
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        head = [parameter for name, parameter in core_model.named_parameters() if not name.startswith("backbone")]
        backbone = [parameter for parameter in core_model.backbone.parameters() if parameter.requires_grad]
        groups = [{"params": head, "lr": 3e-4 if phase == "head" else 5e-5}]
        if backbone:
            groups.append({"params": backbone, "lr": 2e-6})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.05)
        steps = math.ceil(len(train_paths) / 32) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, max(1, round(0.05 * steps)), steps
        )
        for epoch in range(epochs):
            global_epoch += 1
            model.train()
            running, seen = 0.0, 0
            train_loader = loader(
                train_paths, train_labels, training=True,
                epoch_seed=SEED + 100 * global_epoch,
            )
            for batch, labels, _ in train_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, binary_logits, _ = model(batch)
                    class_loss = F.cross_entropy(
                        logits, labels, weight=class_weights, label_smoothing=0.05
                    )
                    pair = labels != 1
                    binary_loss = F.binary_cross_entropy_with_logits(
                        binary_logits[pair], (labels[pair] == 0).float()
                    )
                    loss = class_loss + 0.20 * binary_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running += float(loss.detach()) * len(labels)
                seen += len(labels)
            valid_logits, valid_binary = predict(validation_loader)
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "epoch": epoch + 1,
                "train_loss": running / seen,
                **metrics(valid_labels, hierarchical_probability(valid_logits, valid_binary, 0.8)),
            }
            history.append(row)
            print(json.dumps({"naflex": row}), flush=True)
            if row["macro_f1"] > best_score:
                best_score = row["macro_f1"]
                torch.save({key: value.detach().cpu() for key, value in core_model.state_dict().items()}, checkpoint)

    core_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    for parameter in core_model.backbone.parameters():
        parameter.requires_grad = False
    for block in core_model.backbone.vision_model.encoder.layers[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for layer in (
        core_model.backbone.vision_model.post_layernorm,
        core_model.backbone.vision_model.head,
        core_model.classifier,
        core_model.binary,
    ):
        for parameter in layer.parameters():
            parameter.requires_grad = True
    head = [parameter for name, parameter in core_model.named_parameters() if not name.startswith("backbone")]
    backbone = [parameter for parameter in core_model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": head, "lr": 1e-5}, {"params": backbone, "lr": 5e-7}],
        weight_decay=0.05,
    )
    audit_weights = np.where(
        np.isin(manifest.filename.to_numpy()[train_mask], list(AUDIT_FILENAMES)),
        1.5,
        1.0,
    ).astype(np.float32)
    audit_loader = loader(
        train_paths,
        train_labels,
        training=True,
        weights=audit_weights,
        epoch_seed=SEED + 900,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, round(0.05 * len(audit_loader))), len(audit_loader)
    )
    model.train()
    running, seen = 0.0, 0
    for batch, labels, sample_weights in audit_loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        labels = labels.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, binary_logits, _ = model(batch)
            class_loss = F.cross_entropy(
                logits, labels, weight=class_weights, label_smoothing=0.05
            )
            pair = labels != 1
            binary_rows = F.binary_cross_entropy_with_logits(
                binary_logits[pair], (labels[pair] == 0).float(), reduction="none"
            )
            binary_weights = sample_weights[pair]
            binary_loss = (binary_rows * binary_weights).sum() / binary_weights.sum().clamp_min(1.0)
            loss = class_loss + 0.20 * binary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += float(loss.detach()) * len(labels)
        seen += len(labels)
    valid_logits, valid_binary = predict(validation_loader)
    history.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "audited_material_continuation",
            "epoch": 1,
            "train_loss": running / seen,
            "audited_train_rows": int(np.sum(audit_weights > 1)),
            **metrics(valid_labels, hierarchical_probability(valid_logits, valid_binary, 0.5)),
        }
    )
    torch.save({key: value.detach().cpu() for key, value in core_model.state_dict().items()}, Path(work_dir) / "naflex_audited.pt")
    if checkpoint_commit is not None:
        checkpoint_commit()
    train_features = predict(loader(manifest.path), return_features=True)[2]
    inference_features = predict(loader(inference_paths), return_features=True)[2]
    del model, core_model
    torch.cuda.empty_cache()
    return (
        train_features.astype(np.float16).astype(np.float32),
        inference_features.astype(np.float16).astype(np.float32),
        history,
    )


def logistic(c):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=c,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        tol=1e-5,
        random_state=SEED,
    )

def write_manifest(path, manifest, labels):
    output = manifest[["path", "group"]].copy()
    output["label"] = labels
    output[["path", "label", "group"]].to_csv(path, index=False)


def run_pipeline(data_root, output_dir, prepared_dataset=None, checkpoint_commit=None):
    from datetime import datetime, timezone
    import platform

    import joblib
    import sklearn
    import torch
    import transformers

    started_at = datetime.now(timezone.utc)
    seed_everything()
    if torch.cuda.device_count() < 1:
        raise RuntimeError("This recipe requires one CUDA GPU.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if prepared_dataset is None:
        manifest, inference_paths = discover_dataset(data_root)
    else:
        manifest, inference_paths = prepared_dataset
    feature_labels, final_labels, swap_audit = build_manifests(manifest)
    folds = stable_folds(manifest)
    write_manifest(output_dir / "manifest_audited_final.csv", manifest, final_labels)
    swap_audit.to_csv(output_dir / "manifest_changes.csv", index=False)

    naflex_train, naflex_inference, training_history = train_extract_naflex(
        manifest, inference_paths, feature_labels, folds, "cuda:0", output_dir,
        checkpoint_commit=checkpoint_commit,
    )
    np.savez_compressed(
        output_dir / "features.npz",
        training=naflex_train.astype(np.float16),
        inference=naflex_inference.astype(np.float16),
        labels=final_labels.astype(np.int8),
        folds=folds.astype(np.int8),
    )
    (output_dir / "training_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in training_history),
        encoding="utf-8",
    )
    train_mask, valid_mask = folds != 0, folds == 0
    from sklearn.metrics import log_loss

    head_rows = []
    validation_by_c = {}
    for c in HEAD_CANDIDATES:
        head = logistic(c).fit(naflex_train[train_mask], final_labels[train_mask])
        probability = head.predict_proba(naflex_train[valid_mask])
        validation_by_c[c] = probability
        head_rows.append({
            "C": c,
            **metrics(final_labels[valid_mask], probability),
            "log_loss": float(log_loss(final_labels[valid_mask], probability, labels=[0, 1, 2])),
        })
    head_ablation = pd.DataFrame(head_rows).sort_values(
        ["errors", "log_loss", "macro_f1", "C"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    head_ablation.insert(0, "validation_rank", np.arange(1, len(head_ablation) + 1))
    head_ablation["selected_by_validation"] = head_ablation.validation_rank.eq(1)
    head_ablation.to_csv(output_dir / "validation_head_ablation.tsv", sep="\t", index=False)
    selected_c = float(head_ablation.iloc[0].C)
    validation_probability = validation_by_c[selected_c]

    final_head = logistic(selected_c).fit(naflex_train, final_labels)
    inference_probability = final_head.predict_proba(naflex_inference)
    joblib.dump(final_head, output_dir / "balanced_lr.joblib")
    inference_ids = np.arange(1, 1_459)
    pd.DataFrame(
        {"id": inference_ids, "predicted": inference_probability.argmax(1)}
    ).to_csv(output_dir / "submission.csv", index=False)
    np.savez_compressed(
        output_dir / "probabilities.npz",
        validation=validation_probability.astype(np.float32),
        inference=inference_probability.astype(np.float32),
        validation_target=final_labels[valid_mask].astype(np.int8),
        inference_ids=inference_ids,
        folds=folds,
    )
    report = {
        "clean_retrain": True,
        "model": {
            "id": NAFLEX_ID,
            "revision": NAFLEX_REVISION,
            "max_num_patches": 256,
            "gpu_count": torch.cuda.device_count(),
        },
        "manifest": {
            "changed_rows": 514,
            "class_counts": pd.Series(final_labels).value_counts().sort_index().to_dict(),
        },
        "primary_validation_target": "final514",
        "head": f"balanced logistic regression C={selected_c:g}; Fold-0 error/log-loss-selected",
        "head_selection_rule": "errors ascending, log_loss ascending, macro_f1 descending, C ascending",
        "model_artifacts": [
            "naflex_best.pt",
            "naflex_audited.pt",
            "balanced_lr.joblib",
            "features.npz",
            "preprocessor_config.json",
        ],
        "calibration": None,
        "validation": metrics(final_labels[valid_mask], validation_probability),
        "training_history": training_history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "seed": SEED,
        "model_id": NAFLEX_ID,
        "model_revision": NAFLEX_REVISION,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    inventory = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        inventory.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    (output_dir / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    checksums = [f'{row["sha256"]}  {row["filename"]}' for row in inventory]
    checksums.append(
        f'{hashlib.sha256((output_dir / "artifact_inventory.json").read_bytes()).hexdigest()}  artifact_inventory.json'
    )
    (output_dir / "artifact_checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return report



import modal

HF_CACHE = "/cache/huggingface"
OUTPUT_ROOT = "/cache/last_hope_final_naflex_audited514_error_logloss_balancedlr_a100_ajeng"
REMOTE_ONLY_ARTIFACTS = {"naflex_best.pt", "naflex_audited.pt", "features.npz"}
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "transformers==4.56.2",
        "huggingface-hub==0.34.4", "safetensors==0.6.2",
        "scikit-learn==1.7.2", "pandas==2.3.2", "numpy==2.2.6", "pillow==11.3.0",
    )
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": str(SEED), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)
data_volume = modal.Volume.from_name("bdc2026-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("bdc2026-model-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App("last-hope-final-naflex-audited514-error-logloss-balancedlr-a100-ajeng")


@app.function(image=image, cpu=1, memory=512, timeout=5 * 60)
def smoke_import():
    return {"standalone": True, "final_swaps": len(FINAL_SWAPS)}



@app.function(
    image=image, gpu="A100-40GB", cpu=12, memory=49152, timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume}, secrets=[hf_secret],
)
def train_clean(force: bool = False):
    import shutil

    output = Path(OUTPUT_ROOT)
    if (output / "metrics.json").is_file() and (output / "probabilities.npz").is_file() and not force:
        report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        return {
            "report": report,
            "remote_root": str(output),
            "files": {
                path.name: path.read_bytes()
                for path in output.iterdir()
                if path.is_file() and path.name not in REMOTE_ONLY_ARTIFACTS
            },
        }
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "pipeline_snapshot.py")
    report = run_pipeline("/data/BDC2026", output, checkpoint_commit=cache_volume.commit)
    cache_volume.commit()
    return {
        "report": report,
        "remote_root": str(output),
        "files": {
            path.name: path.read_bytes()
            for path in output.iterdir()
            if path.is_file() and path.name not in REMOTE_ONLY_ARTIFACTS
        },
    }


@app.local_entrypoint()
def main(force: bool = False, output_dir: str = "experiments/final514/results"):
    result = train_clean.remote(force=force)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in result["files"].items():
        (destination / name).write_bytes(content)
    print(json.dumps(result["report"], indent=2))
    print("Heavy artifacts:", result["remote_root"])
    print("Light artifacts:", destination.resolve())
