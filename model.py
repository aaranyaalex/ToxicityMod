from data import Dataset
from tensorflow import string
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from matplotlib import pyplot as plt
import pandas as pd
from numpy import expand_dims
import pickle

class ToxicModel(Dataset):
    def __init__(self, from_file=None) -> None:

        # Prepares the dataset
        super().__init__(max_features=200000, max_text_length=1800)
        self.prepare()

        if from_file:
            self.model = load_model(from_file)
            self.model_history = None
            #self.get_stats()
        else:
            self.create_model()

    def create_model(self):
        self.model = Sequential()
        # Create the embedding layer 
        self.model.add(Embedding(self.max_features+1, 32))
        # Bidirectional LSTM Layer
        self.model.add(Bidirectional(LSTM(32, activation='tanh')))
        # Feature extractor Fully connected layers
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        # Final layer needs to have length 6 to correspond with the 6 toxicity flags
        self.model.add(Dense(6, activation='sigmoid'))
        self.model.compile(loss='BinaryCrossentropy', optimizer="Adam")

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
        self.precision = Precision()
        self.recall = Recall()
        self.accuracy = CategoricalAccuracy()

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

    def save_model(self, name):
        self.model.save(name + '.h5')
    
    def save_tokenizer(self):
        model = Sequential()
        model.add(Input(shape=(1,), dtype=string))
        model.add(self.tokenizer)

        # Save.
        filepath = "vectorizer"
        model.save(filepath, save_format="h5")
        

if __name__ == "__main__":
    # This is the main model I'll use
    #botModel = ToxicModel()
    #botModel.train_model(epochs=1, plot_history=True)
    #botModel.model.save("AaranyasBotModel.h5")

    #text = botModel.tokenizer("You freaking suck! I am going to hit you.")
    #result = botModel.model.predict(expand_dims(text,0))
    #print(result)
    
    # For 10 epochs         For 1 Epoch
    # Precision: 96%            83%
    # Accuracy: 51%             46%
    # Recall: 95%               67%
    # These values are good - the data is quite skewed so accuracy can be low, but we want to definitely flag the toxic messages
    # High Precision and recall demonstrates that model can identify toxic messages well

    #del botModel

    botModel = ToxicModel(from_file="AaranyasBotModel.h5")
            # Load.
    loaded_model = load_model("vectorizer.h5")
    loaded_vectorizer = loaded_model.layers[0]
    #botModel.score("You're so ugly and you smell so bad")

    #


