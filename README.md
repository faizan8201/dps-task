### Command and Code for end-to-end deployment of ML alogrithm using Vertex Ai

### Vertex AI: 
##### Vertex AI Workbench is the single environment for data scientists to complete all of their ML work, from experimentation, to deployment, to managing and monitoring models. It is a Jupyter-based fully managed, scalable, enterprise-ready compute infrastructure with security controls and user management capabilities. Vertex AI brings together the Google Cloud services for building ML under one, unified UI and API. In Vertex AI, you can now easily train and compare models using AutoML or custom code training and all your models are stored in one central model repository. These models can now be deployed to the same endpoints on Vertex AI.

### Setup and Execution in Vertex AI:
##### The end-to-end process of ML algorithm deployment in Vertex AI can be divided into 5 major sections. Which we will discuss individually along with terminal commands for operations.

#### 1. Environment Setup:
##### First we create a login at google [cloud platform](https://cloud.google.com/gcp?utm_source=google&utm_medium=cpc&utm_campaign=emea-de-all-en-bkws-all-all-trial-e-gcp-1011340&utm_content=text-ad-none-any-DEV_c-CRE_500236788645-ADGP_Hybrid%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20~%20GCP%20~%20General%23v1-KWID_43700060393213373-kwd-6458750523-userloc_9042923&utm_term=KW_google%20cloud-NET_g-PLAC_&gclid=Cj0KCQiApL2QBhC8ARIsAGMm-KHD-VJoRSENNzfDhHIrAMjExGMS4GG1aA0F7L6gPL1wnrdqFt9bQTwaAq2UEALw_wcB&gclsrc=aw.ds), and create a new project alongside its terminal activation. Furthermore, we run the following command in the terminal for authentication purpose.

 ``` 
 gcloud auth list
 ```
 ##### This return a confirmation in the following way
 <img width="750" alt="Screenshot 2022-02-18 at 18 58 48" src="https://user-images.githubusercontent.com/49519053/154736985-68e9784a-7979-4047-acc5-72648fe347fa.png">


##### TO confirm the situation following commands are to be run and the output will be as below.

```
gcloud config list project
```
<img width="761" alt="Screenshot 2022-02-18 at 18 59 59" src="https://user-images.githubusercontent.com/49519053/154737132-2855f34d-49d6-4239-a5f9-5aed0ee0a40b.png">

 
 ##### This command returns a confirmation about the activation, and we move towards enabling APIs that we require for this project.
 
 ```
 gcloud services enable compute.googleapis.com         \
                       containerregistry.googleapis.com  \
                       aiplatform.googleapis.com
 ```
 <img width="937" alt="Screenshot 2022-02-18 at 19 02 28" src="https://user-images.githubusercontent.com/49519053/154737522-d4aa1dbf-b594-489a-98eb-71eec38920f1.png">


##### During training process of the model we get our results in the form of assets and to store these saved models we need a bucket inside our project. So to generate a bucket to store assets, we run the following commands inside the shell and get our job done with the bucket creation.

``` 
BUCKET_MPG = gs://$GOOGLE_CLOUD_PROJECT-bucket

gsutil mb -l us-central1 $BUCKET_MPG
```

##### To ensure we use Python3 in our lab
```
alias python=python3
```

#### 2. Containerizing the Code:

##### In this section first, we will create our project files along with the training code. And then build an image out of these files and finally push them to our Google Container Registry

##### To create the folders for our code, we create the directories inside our project along with a docker file, that will help us in container creation. To execute the process, the following commands are run

```
mkdir mpg               # create a directory named mpg
cd mpg                  # navigate to mpg directory
touch Dockerfile        # add a docker file inside mpg
mkdir trainer           # create a new directory inside mpg
touch trainer/train.py 
```

<img width="715" alt="Screenshot 2022-02-18 at 18 20 08" src="https://user-images.githubusercontent.com/49519053/154731358-0c65bc68-c21d-467b-96ae-9f4045eae568.png">


##### In the next step, we add all the necessary commands required by the docker file to run the image. In our case, we will use Deep Learning Container TensorFlow Enterprise 2.3 Docker image, that contains several ML frameworks that we need for the project. After downloading the image, the docker file defines the entry point of the training code. To execute the functionality, we navigate to Docker file and write down the following code and save it.
```
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]

```


##### Now, we add the code to our ```train.py``` file, this code includes all the functions, operations and methods to load, clean, scale the data along with model training and saving the model upon optimisation of the weights. We navigate to the mpg/trainer/train.py and write down the following code and save it.

```

BUCKET = "BUCKET_MPG"

import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers



"""## The Auto MPG dataset

The dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/).

### Get the data
First download the dataset.
"""

"""Import it using pandas"""

dataset_path = "https://storage.googleapis.com/io-vertex-codelab/auto-mpg.csv"
dataset = pd.read_csv(dataset_path, na_values = "?")

dataset.tail()

"""### Clean the data

The dataset contains a few unknown values.
"""

dataset.isna().sum()

"""To keep this initial tutorial simple drop those rows."""

dataset['horsepower'].fillna((dataset['horsepower'].mean()), inplace=True)


"""The `"origin"` column is really categorical, not numeric. So convert that to a one-hot:"""

dataset['origin'] = dataset['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})


dataset = pd.get_dummies(dataset,columns=['origin', 'cylinders', 'model year'] , prefix='', prefix_sep='').drop('car name', axis=1)

dataset.tail()

"""### Split the data into train and test

Now split the dataset into a training set and a test set.

We will use the test set in the final evaluation of our model.
"""

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

"""### Inspect the data

Have a quick look at the joint distribution of a few pairs of columns from the training set.

Also look at the overall statistics:
"""

train_stats = train_dataset.describe()
train_stats.pop("mpg")
train_stats = train_stats.transpose()
train_stats

"""### Split features from labels

Separate the target value, or "label", from the features. This label is the value that you will train the model to predict.
"""

train_labels = train_dataset.pop('mpg')
test_labels = test_dataset.pop('mpg')

"""### Normalize the data

Look again at the `train_stats` block above and note how different the ranges of each feature are.

It is good practice to normalize features that use different scales and ranges. Although the model *might* converge without feature normalization, it makes training more difficult, and it makes the resulting model dependent on the choice of units used in the input.

Note: Although we intentionally generate these statistics from only the training dataset, these statistics will also be used to normalize the test dataset. We need to do that to project the test dataset into the same distribution that the model has been trained on.
"""

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

"""This normalized data is what we will use to train the model.

Caution: The statistics used to normalize the inputs here (mean and standard deviation) need to be applied to any other data that is fed to the model, along with the one-hot encoding that we did earlier.  That includes the test set as well as live data when the model is used in production.

## The model

### Build the model

Let's build our model. Here, we'll use a `Sequential` model with two densely connected hidden layers, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, `build_model`, since we'll create a second model, later on.
"""

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

model = build_model()

"""### Inspect the model

Use the `.summary` method to print a simple description of the model
"""

model.summary()

"""Now try out the model. Take a batch of `10` examples from the training data and call `model.predict` on it.

It seems to be working, and it produces a result of the expected shape and type.

### Train the model

Train the model for 1000 epochs, and record the training and validation accuracy in the `history` object.

Visualize the model's training progress using the stats stored in the `history` object.

This graph shows little improvement, or even degradation in the validation error after about 100 epochs. Let's update the `model.fit` call to automatically stop training when the validation score doesn't improve. We'll use an *EarlyStopping callback* that tests a training condition for  every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training.

You can learn more about this callback [here](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
"""

model = build_model()

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, 
                    callbacks=[early_stop])


# Export model and save to GCS
model.save(BUCKET + '/mpg/model')
```

##### Now, We move back to the shell/terminal and define a variable with the name of our image URL and the build the container inside the root of mpg directory

```
# variable with image url
IMAGE_URI="gcr.io/$GOOGLE_CLOUD_PROJECT/mpg:v1"

```
##### Run this in inside mpg root to build the container

```
docker build ./ -t $IMAGE_URI

```

<img width="822" alt="Screenshot 2022-02-18 at 19 20 05" src="https://user-images.githubusercontent.com/49519053/154740266-56bcfb00-bfde-4f53-a354-d5c09e0f77b2.png">

##### Once the container is build, we push it to our Google Container Registry

```
docker push $IMAGE_URI
```

##### To verify the push, we must locate an image named mpg inside the **Container Registry** section 

<img width="715" alt="Screenshot 2022-02-18 at 18 21 24" src="https://user-images.githubusercontent.com/49519053/154731656-cbc81ed8-1495-4f04-8115-19d0f9ed101d.png">


#### 3. Training on Vertex AI:

##### After successful compilation of image creation and pushing it to the Containers Registry, we navigate to the Training section in the Vertex section of Cloud console. There we create new training by selecting the following parameters for setting up the process pipeline.

>Under Dataset, select No managed dataset

> Then select Custom training (advanced) as your training method and click Continue.

> Enter mpg for Model name

##### Click Continue

> In the Container settings step, select Custom container:

> In the first box (Container image), click Browse and find the container you just pushed to Container Registry. It should look something like this:

<img width="562" alt="Screenshot 2022-02-18 at 19 25 47" src="https://user-images.githubusercontent.com/49519053/154741591-a61d0855-c6c7-4bd2-87e1-3611d6e9f370.png">


##### Next, In container settings we go for Custom conatianer and Browse the image we added to the Registry. Then we select the GPU/CPU as per our computation complexity, in our case it is a simple example, so we will go for the minimum powered machine. 

##### Up next, Under the Prediction container step, select No prediction container and move to the next Section. And we are ready to deploy our model


#### 4. Deploy model end-point:

##### In this step we'll create an endpoint for our trained model. We can use this to get predictions on our model via the Vertex AI API. Here we'll be using the Vertex AI SDK to create a model, deploy it to an endpoint, and get a prediction.

#### To install SDK, navigate to Cloud Shell terminal, and run the following to install the Vertex AI SDK:

```
pip3 install google-cloud-aiplatform --upgrade --use
```


<img width="1274" alt="Screenshot 2022-02-18 at 19 38 07" src="https://user-images.githubusercontent.com/49519053/154742810-0aceaef4-f725-4e08-83db-527495ee0876.png">


Now, We'll create a Python file ```deploy.py``` inside Cloud Shell Editor (create a new file named deploy.py) and use the SDK to create a model resource and deploy it to an endpoint.

<img width="715" alt="Screenshot 2022-02-18 at 18 23 45" src="https://user-images.githubusercontent.com/49519053/154732204-c5132d79-25f9-4636-b66d-822e6c8e7b72.png">

And paste the code as written below inside the python file.

```
from google.cloud import aiplatform

# Create a model resource from public model assets
model = aiplatform.Model.upload(
    display_name="mpg-imported",
    artifact_uri="gs://io-vertex-codelab/mpg-model/",
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-3:latest"
)

# Deploy the above model to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4"
)

```
Then we navigate back to Cloud Shell Terminal and take a step back to root directory with ```cd``` command and run the python script we just created with the following command

```
# navigate back to root
cd ..
# run the deploy.py script
python3 deploy.py | tee deploy-output.txt
```

##### If, we navigate to Model Section on the console in Vertex AI, there will be an end point created inside 'mpg-imported'. This process takes upto 10 minutes to finish. And end-point will be ready.

<img width="1011" alt="Screenshot 2022-02-18 at 19 43 41" src="https://user-images.githubusercontent.com/49519053/154743674-b8c23ae6-9402-41f4-a214-e7ca17e6edf5.png">

##### On clicking 'mpg-imported' we can see the progresso of Deployment and as mentioned earlier it takes up 10 minutes to finish. And we will se something like this once it ends it will result into an end-point

<img width="991" alt="Screenshot 2022-02-18 at 19 51 53" src="https://user-images.githubusercontent.com/49519053/154744549-df6196ff-39d6-4ec8-9f04-5cd4e0ec566b.png">


##### Now, we create a new python file named ```predict.py``` to predict the outcome for our testing values. This file is also created from Cloud Shell Editor and we paste the following code to save it.

```
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="ENDPOINT_MPG"
)

# A test example we'll send to our model for prediction
test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0]

response = endpoint.predict([test_mpg])

print('API response: ', response)

print('Predicted MPG: ', response.predictions[0][0])
```
##### To provide our own endpoint id we navigate to terminal/shell and run the following command
```
ENDPOINT=$(cat deploy-output.txt | sed -nre 's:.*Resource name\: (.*):\1:p' | tail -1)
sed -i "s|ENDPOINT_MPG|$ENDPOINT|g" predict.py
```
##### Finally, we have completed  all the steps and it's time to run the ```predict.py``` to observe the final outcome of the model.

```
python3 predict.py
```

##### Running this will result the output for the model we deployed for our custom independent variables.

<img width="1172" alt="Screenshot 2022-02-18 at 20 08 58" src="https://user-images.githubusercontent.com/49519053/154746792-14148750-5b59-456e-bf34-f214ba109903.png">


#### Sources:
https://cloud.google.com/vertex-ai
