# 用 Keras 构建神经网络

幸运的是，每次我们需要使用神经网络时，都不需要编写激活函数、梯度下降等。有很多包可以帮助我们，建议你了解这些包，包括以下包：

Keras:https://keras.io/

TensorFlow:https://www.tensorflow.org/

Caffe:http://caffe.berkeleyvision.org/

Theano:http://deeplearning.net/software/theano/

Scikit-learn:https://scikit-learn.org/

以及很多其他包！

Keras 使神经网络的编写过程更简单。为了展示有多简单，你将用几行代码构建一个完全连接的简单网络。


用 Keras 构建神经网络
要使用 Keras，你需要知道以下几个核心概念。

## 序列模型

keras.models.Sequential 类是神经网络模型的封装容器。它会提供常见的函数，例如 fit()、evaluate() 和 compile()。


```python
from keras.models import Sequential

#Create the Sequential model
model = Sequential()
```

层
Keras 层就像神经网络层。有全连接层、最大池化层和激活层。你可以使用模型的 add() 函数添加层。例如，简单的模型可以如下所示：


```python
 from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

#创建序列模型
model = Sequential()

#第一层 - 添加有128个节点的全连接层以及32个节点的输入层
model.add(Dense(128, input_dim=32))

#第二层 - 添加 softmax 激活层
model.add(Activation('softmax'))

#第三层 - 添加全连接层
model.add(Dense(10))

#第四层 - 添加 Sigmoid 激活层
model.add(Activation('sigmoid'))
```

Keras 将根据第一层自动推断后续所有层的形状。这意味着，你只需为第一层设置输入维度。

上面的第一层 model.add(Dense(input_dim=32)) 将维度设为 32（表示数据来自 32 维空间）。第二层级获取第一层级的输出，并将输出维度设为 128 个节点。这种将输出传递给下一层级的链继续下去，直到最后一个层级（即模型的输出）。可以看出输出维度是 10。

构建好模型后，我们就可以用以下命令对其进行编译。我们将损失函数指定为我们一直处理的 categorical_crossentropy。我们还可以指定优化程序，稍后我们将了解这一概念，暂时将使用 adam。最后，我们可以指定评估模型用到的指标。我们将使用准确率。


```python
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])
```


```python
#我们可以使用以下命令来查看模型架构：

model.summary()
```

    Model: "sequential_10"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_13 (Dense)             (None, 128)               4224      
    _________________________________________________________________
    activation_11 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dense_14 (Dense)             (None, 10)                1290      
    _________________________________________________________________
    activation_12 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 5,514
    Trainable params: 5,514
    Non-trainable params: 0
    _________________________________________________________________


然后使用以下命令对其进行拟合，指定 epoch 次数和我们希望在屏幕上显示的信息详细程度。

然后使用fit命令训练模型并通过 epoch 参数来指定训练轮数（周期），每 epoch 完成对整数据集的一次遍历。 verbose 参数可以指定显示训练过程信息类型，这里定义为 0 表示不显示信息。


```python
model.fit(X, y, nb_epoch=1000, verbose=0)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-12-e611094a6613> in <module>
    ----> 1 model.fit(X, y, nb_epoch=1000, verbose=0)
    

    NameError: name 'X' is not defined


注意：在 Keras 1 中，nb_epoch 会设置 epoch 次数，但是在 Keras 2 中，变成了 epochs。

最后，我们可以使用以下命令来评估模型：

model.evaluate()

# 练习

我们从最简单的示例开始。构建一个简单的多层前向反馈神经网络以解决 XOR 问题。

将第一层设为 Dense() 层，并将节点数设为8，且 input_dim 设为 2。

在第二层之后使用 softmax 激活函数。

将输出层节点设为 2，因为输出只有 2 个类别。

在输出层之后使用 softmax 激活函数。
对模型运行 10 个 epoch。


```python
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
# Using TensorFlow 1.0.0; use tf.python_io in later versions
#tf.python.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
# One-hot encoding the output
y = np_utils.to_categorical(y)

# Building the model
xor = Sequential()

# Add required layers
xor.add(Dense(64, input_dim=2))
xor.add(Dense(8))
xor.add(Activation("relu"))
xor.add(Dense(2))
xor.add(Activation("softmax"))



# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric
xor.compile(loss="binary_crossentropy", optimizer="adam", metrics = ['accuracy'])

# Uncomment this line to print the model architecture
xor.summary()

# Fitting the model
history = xor.fit(X, y, nb_epoch=100, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))
```

    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_8 (Dense)              (None, 64)                192       
    _________________________________________________________________
    dense_9 (Dense)              (None, 8)                 520       
    _________________________________________________________________
    activation_7 (Activation)    (None, 8)                 0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 2)                 18        
    _________________________________________________________________
    activation_8 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 730
    Trainable params: 730
    Non-trainable params: 0
    _________________________________________________________________


    /home/leon/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:40: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.


    4/4 [==============================] - 0s 6ms/step
    
    Accuracy:  1.0
    
    Predictions:
    [[0.79214245 0.20785749]
     [0.15228593 0.84771407]
     [0.19008899 0.809911  ]
     [0.7188974  0.28110263]]



```python

```
