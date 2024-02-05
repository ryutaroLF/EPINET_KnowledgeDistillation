# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:54:06 2018

@author: shinyonsei2
"""
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input , Activation
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model#

from matplotlib import pyplot as plt
import numpy as np

np.random.seed(123)

def layer1_multistream1(input_dim1,input_dim2,input_dim3,filt_num):
    i = 1
    seq = Sequential()
    seq.add(Conv2D(int(filt_num),(2,2),input_shape=(input_dim1, input_dim2, input_dim3), padding='valid', name='S1_c1%d' %(i) ))
    seq.add(Activation('relu', name='S1_relu1%d' %(i)))
    seq.add(Conv2D(int(filt_num),(2,2), padding='valid', name='S1_c2%d' %(i) ))
    seq.add(BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
    seq.add(Activation('relu', name='S1_relu2%d' %(i)))
    return seq

def layer1_multistream2(input_dim1,input_dim2,input_dim3,filt_num):
    i = 2
    seq = Sequential()
    seq.add(Conv2D(int(filt_num),(2,2),input_shape=(input_dim1, input_dim2, input_dim3), padding='valid', name='S1_c1%d' %(i) ))
    seq.add(Activation('relu', name='S1_relu1%d' %(i)))
    seq.add(Conv2D(int(filt_num),(2,2), padding='valid', name='S1_c2%d' %(i) ))
    seq.add(BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
    seq.add(Activation('relu', name='S1_relu2%d' %(i)))
    return seq

def layer1_multistream3(input_dim1,input_dim2,input_dim3,filt_num):
    i = 3
    print(f"input_dim1 : {input_dim1}, input_dim2 : {input_dim2}, filt_num:{filt_num}")
    seq = Sequential()
    seq.add(Conv2D(int(filt_num),(2,2),input_shape=(input_dim1, input_dim2, input_dim3), padding='valid', name='S1_c1%d' %(i) ))
    seq.add(Activation('relu', name='S1_relu1%d' %(i)))
    seq.add(Conv2D(int(filt_num),(2,2), padding='valid', name='S1_c2%d' %(i) ))
    seq.add(BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
    seq.add(Activation('relu', name='S1_relu2%d' %(i)))
    #seq.add(Reshape((input_dim1-2,input_dim2-2,int(filt_num))))
    return seq

def layer2_merged1(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    seq = Sequential()
    i = 1
    seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
    seq.add(Activation('relu', name='S2_relu1%d' %(i)))
    seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i)))
    seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
    seq.add(Activation('relu', name='S2_relu2%d' %(i)))
    return seq

def layer2_merged2(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    seq = Sequential()
    i = 2
    seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
    seq.add(Activation('relu', name='S2_relu1%d' %(i)))
    seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i)))
    seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
    seq.add(Activation('relu', name='S2_relu2%d' %(i)))
    return seq

def layer2_merged3(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    seq = Sequential()
    i = 3
    seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
    seq.add(Activation('relu', name='S2_relu1%d' %(i)))
    seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i)))
    seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
    seq.add(Activation('relu', name='S2_relu2%d' %(i)))
    return seq

def layer2_merged4(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    seq = Sequential()
    i = 4
    seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
    seq.add(Activation('relu', name='S2_relu1%d' %(i)))
    seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i)))
    seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
    seq.add(Activation('relu', name='S2_relu2%d' %(i)))
    return seq

def layer2_merged5(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    seq = Sequential()
    i = 5
    seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
    seq.add(Activation('relu', name='S2_relu1%d' %(i)))
    seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i)))
    seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
    seq.add(Activation('relu', name='S2_relu2%d' %(i)))
    return seq

def layer2_merged6(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    seq = Sequential()
    i = 6
    seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
    seq.add(Activation('relu', name='S2_relu1%d' %(i)))
    seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i)))
    seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
    seq.add(Activation('relu', name='S2_relu2%d' %(i)))
    return seq

def layer2_merged7(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    seq = Sequential()
    i = 7
    seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
    seq.add(Activation('relu', name='S2_relu1%d' %(i)))
    seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i)))
    seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
    seq.add(Activation('relu', name='S2_relu2%d' %(i)))
    return seq


def layer3_last(input_dim1,input_dim2,input_dim3,filt_num):
    ''' last layer : Conv - Relu - Conv '''

    seq = Sequential()

    for i in range(1):
        seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S3_c1%d' %(i) )) # pow(25/23,2)*12*(maybe7?) 43 3
        seq.add(Activation('relu', name='S3_relu1%d' %(i)))

    seq.add(Conv2D(1,(2,2), padding='valid', name='S3_last'))

    return seq

def define_epinet(sz_input,sz_input2,view_n,conv_depth,filt_num,learning_rate):

    ''' 4-Input : Conv - Relu - Conv - BN - Relu '''
    input_stack_90d = Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_90d')
    input_stack_0d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_0d')
    input_stack_45d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_45d')
    input_stack_M45d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_M45d')

    ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu '''
    mid_90d1=layer1_multistream1(sz_input,sz_input2,len(view_n),int(filt_num))(input_stack_0d)
    print("mid_90d1")

    mid_90d2=layer1_multistream2(sz_input-2,sz_input2-2,int(filt_num),int(filt_num))(mid_90d1)
    print("mid_90d2")

    mid_90d3=layer1_multistream3(sz_input-4,sz_input2-4,int(filt_num),int(filt_num))(mid_90d2)
    print("mid_merged")

    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    print("mid_merged")
    mid_merged_1=layer2_merged1(sz_input-6,sz_input2-6,int(1*filt_num),int(1*filt_num),conv_depth)(mid_90d3)
    print(sz_input)
    print(mid_merged_1.shape)
    print("mid_merged_1")
    mid_merged_2=layer2_merged2(sz_input-8,sz_input2-8,int(1*filt_num),int(1*filt_num),conv_depth)(mid_merged_1)
    print(mid_merged_2.shape)
    print("mid_merged_2")
    mid_merged_3=layer2_merged3(sz_input-10,sz_input2-10,int(1*filt_num),int(1*filt_num),conv_depth)(mid_merged_2)
    print(mid_merged_3.shape)
    print("mid_merged_3")
    mid_merged_4=layer2_merged4(sz_input-12,sz_input2-12,int(1*filt_num),int(1*filt_num),conv_depth)(mid_merged_3)
    print(mid_merged_4.shape)
    print("mid_merged_4")
    mid_merged_5=layer2_merged5(sz_input-14,sz_input2-14,int(1*filt_num),int(1*filt_num),conv_depth)(mid_merged_4)
    print(mid_merged_5.shape)
    print("mid_merged_5")
    mid_merged_6=layer2_merged6(sz_input-16,sz_input2-16,int(1*filt_num),int(1*filt_num),conv_depth)(mid_merged_5)
    print(mid_merged_6.shape)
    print("mid_merged_6")
    mid_merged_7=layer2_merged7(sz_input-18,sz_input2-18,int(1*filt_num),int(1*filt_num),conv_depth)(mid_merged_6)
    print(mid_merged_7.shape)
    print("output")
    ''' Last Conv layer : Conv - Relu - Conv '''
    output=layer3_last(sz_input-20,sz_input2-20,int(1*filt_num),int(1*filt_num))(mid_merged_7)
    print("model_512")
    #model_512 = Model(inputs = [input_stack_90d,input_stack_0d], outputs = [output])
    model_512 = Model(inputs = [input_stack_90d,input_stack_0d,
                               input_stack_45d,input_stack_M45d], outputs = [output])


    opt = RMSprop(lr=learning_rate)
    model_512.compile(optimizer=opt, loss='mae')
    model_512.summary()

    return model_512
