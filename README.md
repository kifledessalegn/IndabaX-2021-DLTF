# Zero to Mastery Deep Learning with TensorFlow

All of the course materials for the [Zero to Mastery Deep Learning with TensorFlow course](https://dbourke.link/ZTMTFcourse).

This course will teach you foundations of deep learning and TensorFlow as well as prepare you to pass the TensorFlow Developer Certification exam (optional).

## Important links
* 🎥 Watch the [first 14-hours of the course on YouTube](https://dbourke.link/tfpart1part2) (notebooks 00, 01, 02)
* 📖 Read the [beautiful online book version of the course](https://dev.mrdbourke.com/tensorflow-deep-learning/)
* 💻 [Sign up](https://dbourke.link/ZTMTFcourse) to the full course on the Zero to Mastery Academy (videos for notebooks 03-10)
* 🤔 Got questions about the course? Check out the [livestream Q&A for the course launch](https://youtu.be/rqAqcFcfeK8)

## Contents of this page
- [Course materials](https://github.com/mrdbourke/tensorflow-deep-learning#course-materials) (everything you'll need for completing the course)
- [Course structure](https://github.com/mrdbourke/tensorflow-deep-learning#course-structure) (how this course is taught)
- [Should you do this course?](https://github.com/mrdbourke/tensorflow-deep-learning#should-you-do-this-course) (decide by answering a couple simple questions)
- [Prerequisites](https://github.com/mrdbourke/tensorflow-deep-learning#prerequisites) (what skills you'll need to do this course)
- [Exercises & Extra-curriculum](https://github.com/mrdbourke/tensorflow-deep-learning#-exercises---extra-curriculum) (challenges to practice what you've learned and resources to learn more)
- [Ask a question](https://github.com/mrdbourke/tensorflow-deep-learning#ask-questions) (like to know more? go here)
- [Status](https://github.com/mrdbourke/tensorflow-deep-learning#status)
- [Log](https://github.com/mrdbourke/tensorflow-deep-learning#log) (updates, changes and progress)

## Course materials

This table is the ground truth for course materials. All the links you need for everything will be here.

Key:
* **Number:** The number of the target notebook (this may not match the video section of the course but it ties together all of the materials in the table)
* **Notebook:** The notebook for a particular module with lots of code and text annotations (notebooks from the videos are based on these)
* **Data/model:** Links to datasets/pre-trained models for the assosciated notebook
* **Exercises & Extra-curriculum:** Each module comes with a set of exercises and extra-curriculum to help practice your skills and learn more, I suggest going through these **before** you move onto the next module
* **Slides:** Although we focus on writing TensorFlow code, we sometimes use pretty slides to describe different concepts, you'll find them here

**Note:** You can get all of the notebook code created during the videos in the [`video_notebooks`](https://github.com/mrdbourke/tensorflow-deep-learning/tree/main/video_notebooks) directory.

| Number | Notebook | Data/Model | Exercises & Extra-curriculum | Slides |
| ----- |  ----- |  ----- |  ----- |  ----- |
| 00 | [TensorFlow Fundamentals](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/00_tensorflow_fundamentals.ipynb) |  | [Go to exercises & extra-curriculum](https://github.com/mrdbourke/tensorflow-deep-learning#-00-tensorflow-fundamentals-exercises) | [Go to slides](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/slides/00_introduction_to_tensorflow_and_deep_learning.pdf) |
| 01 | [TensorFlow Regression](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/01_neural_network_regression_in_tensorflow.ipynb) |  | [Go to exercises & extra-curriculum](https://github.com/mrdbourke/tensorflow-deep-learning#-01-neural-network-regression-with-tensorflow-exercises) | [Go to slides](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/slides/01_neural_network_regression_with_tensorflow.pdf) |
| 02 | [TensorFlow Classification](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/02_neural_network_classification_in_tensorflow.ipynb) |  | [Go to exercises & extra-curriculum](https://github.com/mrdbourke/tensorflow-deep-learning#-02-neural-network-classification-with-tensorflow-exercises) | [Go to slides](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/slides/02_neural_network_classification_with_tensorflow.pdf) |

## Course structure

This course is code first. The goal is to get you writing deep learning code as soon as possible.

It is taught with the following mantra:

```
Code -> Concept -> Code -> Concept -> Code -> Concept
```

This means we write code first then step through the concepts behind it.

If you've got 6-months experience writing Python code and a willingness to learn (most important), you'll be able to do the course.

## Should you do this course?

> Do you have 1+ years experience with deep learning and writing TensorFlow code?

If yes, no you shouldn't, use your skills to build something. 

If no, move onto the next question.

> Have you done at least one beginner machine learning course and would like to learn about deep learning/pass the TensorFlow Developer Certification?

If yes, this course is for you.

If no, go and do a beginner machine learning course and if you decide you want to learn TensorFlow, this page will still be here.

## Prerequisites

> What do I need to know to go through this course?

* **6+ months writing Python code.** Can you write a Python function which accepts and uses parameters? That’s good enough. If you don’t know what that means, spend another month or two writing Python code and then come back here.
* **At least one beginner machine learning course.** Are you familiar with the idea of training, validation and test sets? Do you know what supervised learning is? Have you used pandas, NumPy or Matplotlib before? If no to any of these, I’d going through at least one machine learning course which teaches these first and then coming back. 
* **Comfortable using Google Colab/Jupyter Notebooks.** This course uses Google Colab throughout. If you have never used Google Colab before, it works very similar to Jupyter Notebooks with a few extra features. If you’re not familiar with Google Colab notebooks, I’d suggest going through the Introduction to Google Colab notebook.
* **Plug:** The [Zero to Mastery beginner-friendly machine learning course](https://dbourke.link/ZTMMLcourse) (I also teach this) teaches all of the above (and this course is designed as a follow on).

## 🛠 Exercises & 📖 Extra-curriculum

To prevent the course from being 100+ hours (deep learning is a broad field), various external resources for different sections are recommended to puruse under your own discrestion.

You can find solutions to the exercises in [`extras/solutions/`](https://github.com/mrdbourke/tensorflow-deep-learning/tree/main/extras/solutions), there's a notebook per set of exercises (one for 00, 01, 02... etc). Thank you to [Ashik Shafi](https://github.com/ashikshafi08) for all of the efforts creating these.

---

### 🛠 00. TensorFlow Fundamentals Exercises

1. Create a vector, scalar, matrix and tensor with values of your choosing using `tf.constant()`.
2. Find the shape, rank and size of the tensors you created in 1.
3. Create two tensors containing random values between 0 and 1 with shape `[5, 300]`.
4. Multiply the two tensors you created in 3 using matrix multiplication.
5. Multiply the two tensors you created in 3 using dot product.
6. Create a tensor with random values between 0 and 1 with shape `[224, 224, 3]`.
7. Find the min and max values of the tensor you created in 6 along the first axis.
8. Created a tensor with random values of shape `[1, 224, 224, 3]` then squeeze it to change the shape to `[224, 224, 3]`.
9. Create a tensor with shape `[10]` using your own choice of values, then find the index which has the maximum value.
10. One-hot encode the tensor you created in 9.

### 📖 00. TensorFlow Fundamentals Extra-curriculum 

* Read through the [list of TensorFlow Python APIs](https://www.tensorflow.org/api_docs/python/), pick one we haven't gone through in this notebook, reverse engineer it (write out the documentation code for yourself) and figure out what it does.
* Try to create a series of tensor functions to calculate your most recent grocery bill (it's okay if you don't use the names of the items, just the price in numerical form).
  * How would you calculate your grocery bill for the month and for the year using tensors?
* Go through the [TensorFlow 2.x quick start for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner) tutorial (be sure to type out all of the code yourself, even if you don't understand it).
  * Are there any functions we used in here that match what's used in there? Which are the same? Which haven't you seen before?
* Watch the video ["What's a tensor?"](https://www.youtube.com/watch?v=f5liqUk0ZTw) - a great visual introduction to many of the concepts we've covered in this notebook.

---

### 🛠 01. Neural network regression with TensorFlow Exercises

1. Create your own regression dataset (or make the one we created in "Create data to view and fit" bigger) and build fit a model to it.
2. Try building a neural network with 4 Dense layers and fitting it to your own regression dataset, how does it perform?
3. Try and improve the results we got on the insurance dataset, some things you might want to try include:
  * Building a larger model (how does one with 4 dense layers go?).
  * Increasing the number of units in each layer.
  * Lookup the documentation of [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) and find out what the first parameter is, what happens if you increase it by 10x?
  * What happens if you train for longer (say 300 epochs instead of 200)? 
4. Import the [Boston pricing dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/boston_housing/load_data) from TensorFlow [`tf.keras.datasets`](https://www.tensorflow.org/api_docs/python/tf/keras/datasets) and model it.

### 📖 01. Neural network regression with TensorFlow Extra-curriculum

* [MIT introduction deep learning lecture 1](https://youtu.be/njKP3FqW3Sk) - gives a great overview of what's happening behind all of the code we're running.
* Reading: 1-hour of [Chapter 1 of Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) by Michael Nielson - a great in-depth and hands-on example of the intuition behind neural networks.
* To practice your regression modelling with TensorFlow, I'd also encourage you to look through [Lion Bridge's collection of datasets](https://lionbridge.ai/datasets/) or [Kaggle's datasets](https://www.kaggle.com/data), find a regression dataset which sparks your interest and try to model.

---

### 🛠 02. Neural network classification with TensorFlow Exercises

1. Play with neural networks in the [TensorFlow Playground](https://playground.tensorflow.org/) for 10-minutes. Especially try different values of the learning, what happens when you decrease it? What happens when you increase it?
2. Replicate the model pictured in the [TensorFlow Playground diagram](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.001&regularizationRate=0&noise=0&networkShape=6,6,6,6,6&seed=0.51287&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&regularization_hide=true&discretize_hide=true&regularizationRate_hide=true&percTrainData_hide=true&dataset_hide=true&problem_hide=true&noise_hide=true&batchSize_hide=true) below using TensorFlow code. Compile it using the Adam optimizer, binary crossentropy loss and accuracy metric. Once it's compiled check a summary of the model.
![tensorflow playground example neural network](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/02-tensorflow-playground-replication-exercise.png)
*Try this network out for yourself on the [TensorFlow Playground website](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.001&regularizationRate=0&noise=0&networkShape=6,6,6,6,6&seed=0.51287&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&regularization_hide=true&discretize_hide=true&regularizationRate_hide=true&percTrainData_hide=true&dataset_hide=true&problem_hide=true&noise_hide=true&batchSize_hide=true). Hint: there are 5 hidden layers but the output layer isn't pictured, you'll have to decide what the output layer should be based on the input data.*
3. Create a classification dataset using Scikit-Learn's [`make_moons()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) function, visualize it and then build a model to fit it at over 85% accuracy.
4. Train a model to get 88%+ accuracy on the fashion MNIST test set. Plot a confusion matrix to see the results after.
5. Recreate [TensorFlow's](https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax) [softmax activation function](https://en.wikipedia.org/wiki/Softmax_function) in your own code. Make sure it can accept a tensor and return that tensor after having the softmax function applied to it.
6. Create a function (or write code) to visualize multiple image predictions for the fashion MNIST at the same time. Plot at least three different images and their prediciton labels at the same time. Hint: see the [classifcation tutorial in the TensorFlow documentation](https://www.tensorflow.org/tutorials/keras/classification) for ideas.
7. Make a function to show an image of a certain class of the fashion MNIST dataset and make a prediction on it. For example, plot 3 images of the `T-shirt` class with their predictions.

### 📖 02. Neural network classification with TensorFlow Extra-curriculum

* Watch 3Blue1Brown's neural networks video 2: [*Gradient descent, how neural networks learn*](https://www.youtube.com/watch?v=IHZwWFHWa-w). After you're done, write 100 words about what you've learned.
  * If you haven't already, watch video 1: [*But what is a Neural Network?*](https://youtu.be/aircAruvnKk). Note the activation function they talk about at the end.
* Watch [MIT's introduction to deep learning lecture 1](https://youtu.be/njKP3FqW3Sk) (if you haven't already) to get an idea of the concepts behind using linear and non-linear functions.
* Spend 1-hour reading [Michael Nielsen's Neural Networks and Deep Learning book](http://neuralnetworksanddeeplearning.com/index.html).
* Read the [ML-Glossary documentation on activation functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html). Which one is your favourite?
  * After you've read the ML-Glossary, see which activation functions are available in TensorFlow by searching "tensorflow activation functions".

---

