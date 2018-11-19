
import sys





from classification import classifier

from models.frontend.vgg import VGG

class_number = 5
model = VGG(input_shape=[512,512,3])()

layer_shape = [1000,1000,5]
activation = ['relu','relu','relu']

model.summary()

model = classifier(model,layer_shape=layer_shape,activation=activation)

model.summary()