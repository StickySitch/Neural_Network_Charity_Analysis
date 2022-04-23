#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import tensorflow as tf

#  Importing and reading the charity_data.csv
import pandas as pd
application_df = pd.read_csv("Resources/charity_data.csv")
application_df.head()


# In[2]:


# Dropping the ID columns, 'EIN' and 'NAME'.
application_df = application_df.drop(columns=["EIN", "NAME"])
application_df.head()


# In[3]:


# Determining the number of unique values in each column.
application_df.nunique()


# In[4]:


# Looking at APPLICATION_TYPE value counts for binning
app_type_counts = application_df.APPLICATION_TYPE.value_counts()
app_type_counts


# In[5]:


# Visualizing the value counts of APPLICATION_TYPE
app_type_counts.plot.density()


# In[6]:


# Determining which values to replace if counts are less than 200
replace_application = list(app_type_counts[app_type_counts < 200].index)

# Replacing in dataframe
for app in replace_application:
    application_df.APPLICATION_TYPE = application_df.APPLICATION_TYPE.replace(app,"Other")

# Checking to make sure binning was successful
application_df.APPLICATION_TYPE.value_counts()


# In[7]:


# Looking at CLASSIFICATION value counts for binning
classification_counts = application_df.CLASSIFICATION.value_counts()
classification_counts.head(20)


# In[8]:


# Visualizing the value counts of CLASSIFICATION
classification_counts.plot.density()


# In[9]:


# Determining which values to replace if counts are less than 1800
replace_class = list(classification_counts[classification_counts < 1800].index)

# Replacing in dataframe
for cls in replace_class:
    application_df.CLASSIFICATION = application_df.CLASSIFICATION.replace(cls,"Other")

# Checking to make sure binning was successful
application_df.CLASSIFICATION.value_counts()


# In[10]:


# Converting ASK_AMT column data type from int to object for binning
application_df.ASK_AMT = application_df.ASK_AMT.astype(str)

# Looking at ASK_AMT value counts for binning
ask_amt_counts = application_df.ASK_AMT.value_counts()
ask_amt_counts.head(20)


# In[11]:


# Determining which values to replace if counts are less than 25000
replace_ask_amt = list(ask_amt_counts[ask_amt_counts < 25000].index)

# Replacing in dataframe
for amt in replace_ask_amt:
    application_df.ASK_AMT = application_df.ASK_AMT.replace(amt,"Other")

# Checking to make sure binning was successful
application_df.ASK_AMT.value_counts()


# In[12]:


# Generating our categorical variable lists
application_cat = application_df.dtypes[application_df.dtypes == "object"].index.tolist()
application_cat


# In[13]:


# Creating a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fitting and transforming the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(application_df[application_cat]))

# Adding the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names(application_cat)
encode_df.head()


# In[14]:


# Merging one-hot encoded features and dropping the originals
application_df = application_df.merge(encode_df, left_index=True, right_index=True)
application_df = application_df.drop(application_cat,1)
application_df.head()


# In[15]:


# Splitting the preprocessed data into our features and target arrays

y = application_df["IS_SUCCESSFUL"].values
X = application_df.drop("IS_SUCCESSFUL",1).values

# Splitting the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)


# In[16]:


# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# # Deliverable 2: Compile, Train and Evaluate the Model

# In[17]:


len(X_train[0])


# In[18]:


# Defining the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 80
hidden_nodes_layer2 = 30
hidden_nodes_layer3 = 20

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="tanh"))

# Checking the structure of the model
nn.summary()


# In[19]:


# Importing checkpoint dependencies
import os
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the checkpoint path and filenames
os.makedirs("checkpoints/", exist_ok=True)
checkpoint_path = "checkpoints/weights.{epoch:02d}.hdf5"


# In[20]:


# Compiling the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Creating a callback that saves the model’s weights every epoch
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              verbose=1,
                              save_weights_only=True,
                              save_freq="epoch")


# In[21]:


# Training the model
fit_model = nn.fit(X_train, y_train, epochs=100, callbacks=[cp_callback])


# In[22]:


# Evaluating the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# In[23]:


# Exporting the model
nn.save("AlphabetSoupCharity_Optimization.h5")

