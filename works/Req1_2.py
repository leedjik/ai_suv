import tensorflow as tf

class LinearModel(tf.keras.Model):#tf.keras.Model 상속
	def __init__(self, num_units):
		super(LinearModel, self).__init__()
		self.units = num_units
		self.model = tf.keras.layers.Dense(units=self.units)

	def call(self,x):
		return self.model(x)