import os
import json
import re
import sys
import numpy as np
import string
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
import tensorflow.compat.v1.keras.backend as K
import re

AVAILABLE_TFLM_OPS = {
  'AddAbs()',
  'AddAdd()',
  'AddAddN()',
  'AddArgMax()',
  'AddArgMin()',
  'AddAveragePool2D()',
  'AddBatchToSpaceNd()',
  'AddCeil()',
  'AddConcatenation()',
  'AddConv2D()',
  'AddCos()',
  'AddDepthwiseConv2D()',
  'AddDequantize()',
  'AddDetectionPostprocess()',
  'AddDiv()',
  'AddElu()',
  'AddEqual()',
  'AddEthosU()',
  'AddExpandDims()',
  'AddFloor()',
  'AddFullyConnected()',
  'AddGreater()',
  'AddGreaterEqual()',
  'AddHardSwish()',
  'AddL2Normalization()',
  'AddL2Pool2D()',
  'AddLeakyRelu()',
  'AddLess()',
  'AddLessEqual()',
  'AddLog()',
  'AddLogicalAnd()',
  'AddLogicalNot()',
  'AddLogicalOr()',
  'AddLogistic()',
  'AddMaxPool2D()',
  'AddMaximum()',
  'AddMean()',
  'AddMinimum()',
  'AddMul()',
  'AddNeg()',
  'AddNotEqual()',
  'AddPack()',
  'AddPad()',
  'AddPadV2()',
  'AddPrelu()',
  'AddQuantize()',
  'AddReduceMax()',
  'AddRelu()',
  'AddRelu6()',
  'AddReshape()',
  'AddResizeNearestNeighbor()',
  'AddRound()',
  'AddRsqrt()',
  'AddShape()',
  'AddSin()',
  'AddSoftmax()',
  'AddSpaceToBatchNd()',
  'AddSplit()',
  'AddSplitV()',
  'AddSqrt()',
  'AddSquare()',
  'AddSqueeze()',
  'AddStridedSlice()',
  'AddSub()',
  'AddSvdf()',
  'AddTanh()',
  'AddTransposeConv()',
  'AddUnpack()',
}


BUILTIN_OPCODE2NAME = {
    0: 'ADD',
    1: 'AVERAGE_POOL_2D',
    2: 'CONCATENATION',
    3: 'CONV_2D',
    4: 'DEPTHWISE_CONV_2D',
    5: 'DEPTH_TO_SPACE',
    6: 'DEQUANTIZE',
    7: 'EMBEDDING_LOOKUP',
    8: 'FLOOR',
    9: 'FULLY_CONNECTED',
    10: 'HASHTABLE_LOOKUP',
    11: 'L2_NORMALIZATION',
    12: 'L2_POOL_2D',
    13: 'LOCAL_RESPONSE_NORMALIZATION',
    14: 'LOGISTIC',
    15: 'LSH_PROJECTION',
    16: 'LSTM',
    17: 'MAX_POOL_2D',
    18: 'MUL',
    19: 'RELU',
    20: 'RELU_N1_TO_1',
    21: 'RELU6',
    22: 'RESHAPE',
    23: 'RESIZE_BILINEAR',
    24: 'RNN',
    25: 'SOFTMAX',
    26: 'SPACE_TO_DEPTH',
    27: 'SVDF',
    28: 'TANH',
    29: 'CONCAT_EMBEDDINGS',
    30: 'SKIP_GRAM',
    31: 'CALL',
    32: 'CUSTOM',
    33: 'EMBEDDING_LOOKUP_SPARSE',
    34: 'PAD',
    35: 'UNIDIRECTIONAL_SEQUENCE_RNN',
    36: 'GATHER',
    37: 'BATCH_TO_SPACE_ND',
    38: 'SPACE_TO_BATCH_ND',
    39: 'TRANSPOSE',
    40: 'MEAN',
    41: 'SUB',
    42: 'DIV',
    43: 'SQUEEZE',
    44: 'UNIDIRECTIONAL_SEQUENCE_LSTM',
    45: 'STRIDED_SLICE',
    46: 'BIDIRECTIONAL_SEQUENCE_RNN',
    47: 'EXP',
    48: 'TOPK_V2',
    49: 'SPLIT',
    50: 'LOG_SOFTMAX',
    51: 'DELEGATE',
    52: 'BIDIRECTIONAL_SEQUENCE_LSTM',
    53: 'CAST',
    54: 'PRELU',
    55: 'MAXIMUM',
    56: 'ARG_MAX',
    57: 'MINIMUM',
    58: 'LESS',
    59: 'NEG',
    60: 'PADV2',
    61: 'GREATER',
    62: 'GREATER_EQUAL',
    63: 'LESS_EQUAL',
    64: 'SELECT',
    65: 'SLICE',
    66: 'SIN',
    67: 'TRANSPOSE_CONV',
    68: 'SPARSE_TO_DENSE',
    69: 'TILE',
    70: 'EXPAND_DIMS',
    71: 'EQUAL',
    72: 'NOT_EQUAL',
    73: 'LOG',
    74: 'SUM',
    75: 'SQRT',
    76: 'RSQRT',
    77: 'SHAPE',
    78: 'POW',
    79: 'ARG_MIN',
    80: 'FAKE_QUANT',
    81: 'REDUCE_PROD',
    82: 'REDUCE_MAX',
    83: 'PACK',
    84: 'LOGICAL_OR',
    85: 'ONE_HOT',
    86: 'LOGICAL_AND',
    87: 'LOGICAL_NOT',
    88: 'UNPACK',
    89: 'REDUCE_MIN',
    90: 'FLOOR_DIV',
    91: 'REDUCE_ANY',
    92: 'SQUARE',
    93: 'ZEROS_LIKE',
    94: 'FILL',
    95: 'FLOOR_MOD',
    96: 'RANGE',
    97: 'RESIZE_NEAREST_NEIGHBOR',
    98: 'LEAKY_RELU',
    99: 'SQUARED_DIFFERENCE',
    100: 'MIRROR_PAD',
    101: 'ABS',
    102: 'SPLIT_V',
    103: 'UNIQUE',
    104: 'CEIL',
    105: 'REVERSE_V2',
    106: 'Add_N',
    107: 'GATHER_ND',
    108: 'COS',
    109: 'WHERE',
    110: 'RANK',
    111: 'ELU',
    112: 'REVERSE_SEQUENCE',
    113: 'MATRIX_DIAG',
    114: 'QUANTIZE',
    115: 'MATRIX_SET_DIAG',
    116: 'ROUND',
    117: 'HARD_SWISH',
    118: 'IF',
    119: 'WHILE',
    120: 'NON_MAX_SUPPRESSION_V4',
    121: 'NON_MAX_SUPPRESSION_V5',
    122: 'SCATTER_ND',
    123: 'SELECT_V2',
    124: 'DENSIFY',
    125: 'SEGMENT_SUM',
    126: 'BATCH_MATMUL',
    127: 'PLACEHOLDER_FOR_GREATER_OP_CODES',
    128: 'CUMSUM',
}



def CamelCaseToSnakeCase(camel_case_input):
    """Converts an identifier in CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z] )", r"\1_\2", camel_case_input)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def FlatbufferToDict(fb, preserve_as_numpy):
    """Converts a hierarchy of FB objects into a nested dict.
    We avoid transforming big parts of the flat buffer into python arrays. This
    speeds conversion from ten minutes to a few seconds on big graphs.
    Args:
    fb: a flat buffer structure. (i.e. ModelT)
    preserve_as_numpy: true if all downstream np.arrays should be preserved.
    false if all downstream np.array should become python arrays
    Returns:
    A dictionary representing the flatbuffer rather than a flatbuffer object.
    """
    if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
        return fb
    elif hasattr(fb, "__dict__"):
        result = {}
        for attribute_name in dir(fb):
            attribute = fb.__getattribute__(attribute_name)
            if not callable(attribute) and attribute_name[0] != "_":
                snake_name = CamelCaseToSnakeCase(attribute_name)
                preserve = True if attribute_name == "buffers" else preserve_as_numpy
                result[snake_name] = FlatbufferToDict(attribute, preserve)
        return result
    elif isinstance(fb, np.ndarray):
        return fb if preserve_as_numpy else fb.tolist()
    elif hasattr(fb, "__len__"):
        return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
    else:
        return fb


def CreateDictFromFlatbuffer(buffer_data):
    model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    return FlatbufferToDict(model, preserve_as_numpy=False)

  
def get_operator_names_from_tflite_model(tflite_model_dir):
    with open("/home/nesl/swapnil/g_kws_model_data_quant.tflite", "rb") as file_handle:
        file_data = bytearray(file_handle.read())
    data = CreateDictFromFlatbuffer(file_data)

    x = []
    for i in range(len(data["operator_codes"])):
        data["operator_codes"][i]["builtin_code"] = max(
            data["operator_codes"][i]["builtin_code"],
            data["operator_codes"][i]["deprecated_builtin_code"],
        )
        x.append((BUILTIN_OPCODE2NAME[data["operator_codes"][i]["builtin_code"]]))

    return x

  

def tflite_operator_place(x, z): #x is tflite model operator names from above, z is the loaded cpp file
    error_flag = 0
    idx = z.index([i for i in z if 'static tflite::MicroMutableOpResolver' in i][0])
    idx_orig = idx
    idx += 1
    if 'micro_op_resolver' in [i for i in z
                               if 'static tflite::MicroMutableOpResolver'
                                in i][0]:
        resString = 'micro_op_resolver.'
    else:
        resString = 'resolver.'
        
    start_idx = z.index([i for i in z if resString+'Add' in i][0])
    end_idx = z.index([i for i in z if resString+'Add' in i][-1])
    for i in range(start_idx,end_idx+1):
        z[i] = ''
    del(z[start_idx:end_idx+1])

    for i in range(len(x)):
        ss = '_'.join(map(lambda s: s.strip().capitalize(),
                      x[i].split('_'))).translate(str.maketrans('', '',
                string.punctuation))
        if '2' in ss:
            ss = ss[0:-1] + 'D'
        if len([i for i in AVAILABLE_TFLM_OPS if ss in i[3:]]) != 0:
            ins = [i for i in AVAILABLE_TFLM_OPS if ss in i[3:]][0]
            z.insert(idx, resString + ins + ';\n')
            idx += 1
        else:
            print('Error. Model has unsupported operators!')
            error_flag = 1
            break
    if error_flag == 0:
        z[idx_orig] = re.sub("\d+", str(len(x)), [i for i in z if 'static tflite::MicroMutableOpResolver' in i][0])
        if len([i for i in z if 'MicroModelRunner' in i])!=0:
            idx1 = z.index([i for i in z if 'MicroModelRunner' in i][0])
            idx2 = z.index([i for i in z if 'MicroModelRunner' in i][1])
        z[idx1] = re.sub("\d+>", str(len(x))+'>', [i for i in z if z[idx1] in i][0])
        z[idx2] = re.sub("\d+>", str(len(x))+'>', [i for i in z if z[idx2] in i][0])        

    return z, error_flag
        

