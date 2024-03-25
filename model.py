from data import Dataset
from keras import models, layers, losses, metrics
from matplotlib import pyplot as plt
import pandas as pd

class ToxicModel(Dataset):
    def __init__(self, from_file=None) -> None:

        # Prepares the dataset
        super().__init__(max_features=200000, max_text_length=1800)
        self.prepare()

        if from_file:
            self.model = models.load_model(from_file)
            self.model_history = None
            self.get_stats()
        else:
            self.create_model()
            

    def create_model(self):
        self.model = models.Sequential()
        # Create the embedding layer 
        self.model.add(layers.Embedding(self.max_features+1, 32))
        # Bidirectional LSTM Layer
        self.model.add(layers.Bidirectional(layers.LSTM(32, activation='tanh')))
        # Feature extractor Fully connected layers
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        # Final layer needs to have length 6 to correspond with the 6 toxicity flags
        self.model.add(layers.Dense(6, activation='sigmoid'))
        self.model.compile(loss=losses.binary_crossentropy, optimizer="Adam")

        print("Built model skeleton!")

    def train_model(self, epochs=10, plot_history=False):

        print("Beginning training...")
        self.model_history = self.model.fit(self.train, epochs=epochs, validation_data=self.val)
        print("Finished training!")
        if plot_history:
            plt.figure(figsize=(8,5))
            pd.DataFrame(self.model_history.history).plot()
            plt.show()
        
        self.get_stats()
    
    def get_stats(self):
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.accuracy = metrics.CategoricalAccuracy()

        print("Testing the model...")
        for batch in self.test.as_numpy_iterator(): 
            # Unpack the batch 
            x_true, y_true = batch
            # Make a prediction 
            y_pred = self.model.predict(x_true)
            
            # Flatten the predictions
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            
            self.precision.update_state(y_true, y_pred)
            self.recall.update_state(y_true, y_pred)
            self.accuracy.update_state(y_true, y_pred)
        
        print(f'Precision: {self.precision.result().numpy()}, Recall:{self.recall.result().numpy()}, Accuracy:{self.accuracy.result().numpy()}')

    def batch_predict(self, batch):
        pass

    def save_model(self, name):
        self.model.save(name + '.keras')
