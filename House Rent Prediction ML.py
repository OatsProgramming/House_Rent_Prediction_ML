import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from os import system

system('clear')

# VISUALIZERS
def history_visualizer(history):
    pd.DataFrame(history.history).plot()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()

def learning_rate_visualizer(learning_rate, epoch, history):
    lrs = learning_rate * (10 ** (tf.range(epoch)/20))

    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history['loss'])
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.show() 

# MACHINE LEARNING

# Set up

# Set seed for reproducibility
tf.random.set_seed(42)

# Get data
rent_data = pd.read_csv('House_Rent_Dataset.csv')
print('\nRENT DATA\n', rent_data)

# Column transform and onehot encode it
ct = make_column_transformer(
    (MinMaxScaler(), ['BHK', 'Size', 'Bathroom']),
    (OneHotEncoder(handle_unknown='ignore'), ['Posted On', 'Floor', 'Area Type', 'Area Locality', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
)

# Create feature and label
X = rent_data.drop('Rent', axis = 1)
y = rent_data['Rent']

print('\nFEATURE\n', X)
print('\nLABEL\n', y)

# Create training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the column transformer into our data
ct.fit(X_train)

# Transform Training and testing data with normalization
X_train_normalized = ct.transform(X_train)
X_test_normalized = ct.transform(X_test)

print('\nX TRAIN NORMALIZED\n', X_train_normalized)
print('\nX TEST NORMALIZED\n', X_test_normalized)

# The datasets are csr_matrix; convert to numpy arrays to work with model
X_train_normalized = X_train_normalized.toarray()
X_test_normalized = X_test_normalized.toarray()

print('\nX TRAIN SHAPE\n', X_train_normalized.shape)
# Create Compile Fit
LEARNING_RATE = 0.01
EPOCHS = 100

model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(2445, activation = 'relu'),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = 'mae',
    optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE),
    metrics = ['mae']
)

#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * 10**(epoch/20))

history = model.fit(
    X_train_normalized, 
    y_train, 
    epochs = EPOCHS, 
    verbose = 2
    #callbacks = [lr_scheduler]
)

history_visualizer(history)
#learning_rate_visualizer(LEARNING_RATE, EPOCHS, history)