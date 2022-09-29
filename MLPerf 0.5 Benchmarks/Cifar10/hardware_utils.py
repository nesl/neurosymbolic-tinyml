import subprocess
import os
import shutil
import serial
import re
import numpy as np
from operator_name_ex import *
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
from train_utils import *


cifar_10_dir = 'cifar-10-batches-py'

device_list = ["NUCLEO_F746ZG","NUCLEO_L476RG", "NUCLEO_F446RE", "ARCH_MAX", "NUCLEO_L4R5ZI_P", "ISPU_ST"]
maxRAM_list = [280000, 80000,80000,180000,600000, 8000]
maxFlash_list = [800000,800000,400000,400000,1800000, 32000]

class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def get_model_memory_usage(batch_size, model): 
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    bytes_size = (total_memory + internal_model_mem_count)
    return bytes_size

def representative_dataset_generator():
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data(cifar_10_dir)
    _idx = np.load('calibration_samples_idxs.npy')
    for i in _idx:
        sample_img = np.expand_dims(np.array(test_data[i], dtype=np.float32), axis=0)
        yield [sample_img]
        

def convert_to_tflite(model,model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('trained_models/' + model_name + '.tflite', 'wb').write(tflite_model) #floating point

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_generator
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    open('trained_models/' + model_name + '_quant.tflite', 'wb').write(tflite_quant_model) #quantized
    
    
def get_model_flash_usage(model,model_name,quantization=True):
    convert_to_tflite(model,model_name)
    if(quantization==True):
        return os.path.getsize('trained_models/' + model_name + '_quant.tflite')
    else:
        return os.path.getsize('trained_models/' + model_name + '.tflite')
    
def return_hardware_specs(hardware):
    maxRAM = maxRAM_list[device_list.index(hardware)]
    maxFlash = maxFlash_list[device_list.index(hardware)]
    return maxRAM, maxFlash


def convert_to_cpp(model,model_name,quantization=True):
    convert_to_tflite(model,model_name)
    if(quantization==True):
        os.system("xxd -i trained_models/"+ model_name + "_quant.tflite > ic_model_quant_data.cc")
        with open('ic_model_quant_data.cc') as f:
            z = f.readlines()
        f.close()   
        z.insert(0,'#include "ic/ic_model_quant_data.h"\n')
        z = [w.replace('unsigned char','alignas(8) const unsigned char') for w in z]
        z = [w.replace('trained_models_pretrainedResnet_quant_tflite','pretrainedResnet_quant_tflite') for w in z]
        z = [w.replace('unsigned int','const unsigned int') for w in z]
        my_f = open("ic_model_quant_data.cc","w")
        for item in z:
            my_f.write(item)
        my_f.close()
    else:
        os.system("xxd -i trained_models/"+ model_name + ".tflite > ic_model_quant_data.cc")
        with open('ic_model_quant_data.cc') as f:
            z = f.readlines()
        f.close()   
        z.insert(0,'#include "ic/ic_model_quant_data.h"\n')
        z = [w.replace('unsigned char','alignas(8) const unsigned char') for w in z]
        z = [w.replace('trained_models_pretrainedResnet_tflite','pretrainedResnet_quant_tflite') for w in z]
        z = [w.replace('unsigned int','const unsigned int') for w in z]
        my_f = open("ic_model_quant_data.cc","w")
        for item in z:
            my_f.write(item)
        my_f.close()
        
def platform_in_the_loop_controller(model,model_name, hardware,dir_path='Cifar10_Mbed_Prog/',quantization=True):
    err_flag = 0
    RAM = -1
    Flash = -1
    err_flag = -1
    Latency = -1
    ser = serial.Serial('/dev/ttyACM0') #COM Port where board is attached
    convert_to_cpp(model,model_name,quantization)
    shutil.copy('ic_model_quant_data.cc',dir_path+'ic/ic_model_quant_data.cc')
    quant_op_names = get_operator_names_from_tflite_model("trained_models/"+ model_name + "_quant.tflite")
    float_op_names = get_operator_names_from_tflite_model("trained_models/"+ model_name + ".tflite")
    with cd(dir_path):
        with open('main.cpp') as f:
            z = f.readlines()
        f.close()
        z[z.index([i for i in z if 'constexpr int kTensorArenaSize' in i][0])] = 'constexpr int kTensorArenaSize='+str(int(maxRAM_list[device_list.index(hardware)]/1000))+'*1000;\n';
        if(quantization==True):
            z, op_error = tflite_operator_place(quant_op_names, z)
            z[z.index([i for i in z if '> *runner' in i][0])] = 'tflite::MicroModelRunner<int8_t, int8_t, ' + z[z.index([i for i in z if '> *runner' in i][0])][z[z.index([i for i in z if '> *runner' in i][0])].find(re.findall('\d>',z[z.index([i for i in z if '> *runner' in i][0])])[0]):len([i for i in z if '> *runner' in i][0])];
            z[z.index([i for i in z if 'input_asint[kIcInputSize]' in i][0])] = 'int8_t input_asint[kIcInputSize];\n';
            z[z.index([i for i in z if '> model_runner' in i][0])] = 'static tflite::MicroModelRunner<int8_t, int8_t,' + z[z.index([i for i in z if '> model_runner' in i][0])][z[z.index([i for i in z if '> model_runner' in i][0])].find(re.findall('\d>',z[z.index([i for i in z if '> model_runner' in i][0])])[0]):len([i for i in z if '> model_runner' in i][0])];
            z[z.index([i for i in z if 'input_asint[i]' in i][0])] = 'input_asint[i] = (rand() % (255 - 0 + 1)) + 0;\n';
            z[z.index([i for i in z if 'float converted' in i][0])] = 'float converted = DequantizeInt8ToFloat(runner->GetOutput()[i], runner->output_scale(),runner->output_zero_point());\n';
        else:
            z, op_error = tflite_operator_place(float_op_names, z)
            z[z.index([i for i in z if '> *runner' in i][0])] = 'tflite::MicroModelRunner<int8_t, int8_t, ' + z[z.index([i for i in z if '> *runner' in i][0])][z[z.index([i for i in z if '> *runner' in i][0])].find(re.findall('\d>',z[z.index([i for i in z if '> *runner' in i][0])])[0]):len([i for i in z if '> *runner' in i][0])];
            z[z.index([i for i in z if 'input_asint[kIcInputSize]' in i][0])] = 'float input_asint[kIcInputSize];\n';
            z[z.index([i for i in z if '> model_runner' in i][0])] = 'static tflite::MicroModelRunner<float, float, ' +  z[z.index([i for i in z if '> model_runner' in i][0])][z[z.index([i for i in z if '> model_runner' in i][0])].find(re.findall('\d>',z[z.index([i for i in z if '> model_runner' in i][0])])[0]):len([i for i in z if '> model_runner' in i][0])];
            z[z.index([i for i in z if 'input_asint[i]' in i][0])] = 'input_asint[i] = ((float)rand()/(float)(RAND_MAX))*5.0;\n';
            z[z.index([i for i in z if 'float converted' in i][0])] = 'float converted = runner->GetOutput()[i];\n';
        my_f = open("main.cpp","w")
        for item in z:
            my_f.write(item)
        my_f.close()
    
        if(os.path.exists("BUILD/") and os.path.isdir("BUILD/")):
            shutil.rmtree("BUILD/")
        
        if(op_error == 0):
            os.system("mbed config root .")
            os.system("mbed deploy")
            cmd = "mbed compile -m "+ hardware +" -t GCC_ARM --flash"
            sp = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = sp.communicate()
            x = out.decode("utf-8") 
            if(x.find("overflowed")==-1):
                RAM = [float(s) for s in re.findall(r'-?\d+\.?\d*', x[x.find('Static RAM memory'):x.find('Total Flash memory')])][0]
                Flash = [float(s) for s in re.findall(r'-?\d+\.?\d*', x[x.find('Total Flash memory'):x.find('Total Flash memory')+x[x.find('Total Flash memory'):].find('bytes')])][0]                 
                with serial.Serial('/dev/ttyACM0', 9600, timeout=20) as ser:
                    s = ser.read(1000)
                x = s.decode("utf-8") 
                if(x.find("size is too small for all buffers")==-1 and x.find("to allocate")==-1 and x.find("missing")==-1 and x.find("Fault")==-1):
                    Latency = [float(s) for s in re.findall(r'-?\d+\.?\d*', x[x.find('timer output'):x.find('timer output')+x[x.find('timer output'):].find('\n')])][0] 
                    err_flag = 0
                else:
                    err_flag = 1

            elif x.find("counter backwards")!=-1:
                err_flag = 1
            else:
                err_flag = 1
        else:
            err_flag = 1 
    return RAM, Flash, Latency, err_flag
    
    
