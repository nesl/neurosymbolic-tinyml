import subprocess
import os
import shutil
import serial
import re
import numpy as np
from operator_name_ex import *
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
import common as com

param = com.yaml_load()

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

def load_training_data():
    dirs = com.select_dirs(param=param, mode=1)
    idx = 0
    target_dir = dirs[0]
    files = com.file_list_generator(target_dir)
    train_data = com.list_to_vector_array(files,
                                      msg="generate train_dataset",
                                      n_mels=param["feature"]["n_mels"],
                                      frames=param["feature"]["frames"],
                                      n_fft=param["feature"]["n_fft"],
                                      hop_length=param["feature"]["hop_length"],
                                      power=param["feature"]["power"])
    return train_data
    

def convert_to_tflite(model,model_name, train_data):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('trained_models/' + model_name + '.tflite', 'wb').write(tflite_model) #floating point
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    

    def representative_dataset_gen():
        for sample in train_data[::5]:
            sample = np.expand_dims(sample.astype(np.float32), axis=0)
            yield [sample]    
            
    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    open('trained_models/' + model_name + '_quant.tflite', 'wb').write(tflite_quant_model) #quantized
    
    
def get_model_flash_usage(model,model_name,train_data,quantization=True):
    convert_to_tflite(model,model_name,train_data)
    if(quantization==True):
        return os.path.getsize('trained_models/' + model_name + '_quant.tflite')
    else:
        return os.path.getsize('trained_models/' + model_name + '.tflite')
    
def return_hardware_specs(hardware):
    maxRAM = maxRAM_list[device_list.index(hardware)]
    maxFlash = maxFlash_list[device_list.index(hardware)]
    return maxRAM, maxFlash

def convert_to_cpp(model,model_name,train_data,quantization=True):
    convert_to_tflite(model,model_name, train_data)
    if(quantization==True):
        os.system("xxd -i trained_models/"+ model_name + "_quant.tflite > model.cc")
        with open('model.cc') as f:
            z = f.readlines()
        f.close()   
        z.insert(0,'#include "model.h"\n#ifdef __has_attribute\n#define HAVE_ATTRIBUTE(x) __has_attribute(x)\n#else\n#define HAVE_ATTRIBUTE(x) 0\n#endif\n#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))\n#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))\n#else\n#define DATA_ALIGN_ATTRIBUTE\n#endif\n')
        z = [w.replace('unsigned char','const unsigned char') for w in z]
        z = [w.replace('trained_models_g_model_quant_tflite','g_model') for w in z]
        z = [w.replace('const unsigned char g_model[]','const unsigned char g_model[] DATA_ALIGN_ATTRIBUTE') for w in z]
        my_f = open("model.cc","w")
        for item in z:
            my_f.write(item)
        my_f.close()
    else:
        os.system("xxd -i trained_models/"+ model_name + ".tflite > model.cc")
        with open('model.cc') as f:
            z = f.readlines()
        f.close()   
        z.insert(0,'#include "model.h"\n#ifdef __has_attribute\n#define HAVE_ATTRIBUTE(x) __has_attribute(x)\n#else\n#define HAVE_ATTRIBUTE(x) 0\n#endif\n#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))\n#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))\n#else\n#define DATA_ALIGN_ATTRIBUTE\n#endif\n')
        z = [w.replace('unsigned char','const unsigned char') for w in z]
        z = [w.replace('trained_models_g_model_tflite','g_model') for w in z]
        z = [w.replace('const unsigned char g_model[]','const unsigned char g_model[] DATA_ALIGN_ATTRIBUTE') for w in z]
        my_f = open("model.cc","w")
        for item in z:
            my_f.write(item)
        my_f.close()
        
def platform_in_the_loop_controller(model,model_name, hardware,train_data,dir_path='AD_Mbed_Prog/',quantization=True):
    err_flag = 0
    RAM = -1
    Flash = -1
    err_flag = -1
    Latency = -1
    ser = serial.Serial('/dev/ttyACM0') #COM Port where board is attached
    convert_to_cpp(model,model_name,train_data,quantization)
    shutil.copy('model.cc',dir_path+'model.cc')
    shutil.copy('strided_slice.cc',dir_path+'tensorflow/lite/micro/kernels/strided_slice.cc')
    quant_op_names = get_operator_names_from_tflite_model("trained_models/"+ model_name + "_quant.tflite")
    float_op_names = get_operator_names_from_tflite_model("trained_models/"+ model_name + ".tflite")
    with cd(dir_path):
        with open('main.cpp') as f:
            z = f.readlines()
        f.close()
        z[z.index([i for i in z if 'constexpr int kTensorArenaSize' in i][0])] = 'constexpr int kTensorArenaSize='+str(int(maxRAM_list[device_list.index(hardware)]/1000))+'*1000;\n';
        if(quantization==True):
            z, op_error = tflite_operator_place(quant_op_names, z)
            z[z.index([i for i in z if 'model_input_t' in i][0])] = 'typedef int8_t model_input_t;\n'
            z[z.index([i for i in z if 'model_output_t' in i][0])] = 'typedef int8_t model_output_t;;\n'
            z[z.index([i for i in z if '*model_input_buffer' in i][0])] = 'int8_t *model_input_buffer = model_input->data.int8;\n'
            z[z.index([i for i in z if '*feature_buffer_ptr' in i][0])] = 'int8_t *feature_buffer_ptr = input_quantized;\n'
            z[z.index([i for i in z if 'float converted' in i][0])] = '    float converted = DequantizeInt8ToFloat(output->data.int8[i], interpreter->output(0)->params.scale, interpreter->output(0)->params.zero_point);\n'
        else:
            z, op_error = tflite_operator_place(float_op_names, z)
            z[z.index([i for i in z if 'model_input_t' in i][0])] = 'typedef float model_input_t;\n'
            z[z.index([i for i in z if 'model_output_t' in i][0])] = 'typedef float model_output_t;;\n'
            z[z.index([i for i in z if '*model_input_buffer' in i][0])] = 'float *model_input_buffer = model_input->data.float;\n'
            z[z.index([i for i in z if '*feature_buffer_ptr' in i][0])] = 'float *feature_buffer_ptr = input_float;\n'
            z[z.index([i for i in z if 'float converted' in i][0])] = '    float converted = output->data.float[i];\n'
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
    
    
