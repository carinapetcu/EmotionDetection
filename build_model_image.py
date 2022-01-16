import visualkeras
import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Concatenate, BatchNormalization, Dropout
from collections import defaultdict

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[Dense]['fill'] = 'green'
color_map[Dropout]['fill'] = 'pink'
color_map[Input]['fill'] = 'red'
color_map[Flatten]['fill'] = 'teal'
color_map[MaxPool2D]['fill'] = 'violet'
color_map[Concatenate]['fill'] = 'brown'
color_map[BatchNormalization]['fill'] = 'blue'

model = keras.models.load_model('./model_dexpression_based_2.h5')
visualkeras.layered_view(model, color_map=color_map, to_file='model_2_visualization.png').show()

