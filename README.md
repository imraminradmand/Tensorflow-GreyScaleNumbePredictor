# Tensorflow-GreyScaleNumbePredictor
Model that will predict what number is in an image - CNN

# Dataset
Data that the model is being trained, and getting its test data is the MNIST dataset.
Provides enough data for a good train/test split.
Images in the data set are very small only 28x28 and from the name of the repo you can tell they are only grey scale images.

# What model does
From the repo's name you can probably guess it. Give an image of a number between 0 - 9 model can predict what number is in the image.
Pretty simple :)

# Model Metrics

<img width="424" alt="Screen Shot 2020-11-10 at 10 12 01 PM" src="https://user-images.githubusercontent.com/69999501/98771527-e0040a00-23a1-11eb-9055-839af01eb73b.png">
<img width="432" alt="Screen Shot 2020-11-10 at 10 12 07 PM" src="https://user-images.githubusercontent.com/69999501/98771532-e1353700-23a1-11eb-9f49-f6ffde2b1cb6.png">
<img width="469" alt="Screen Shot 2020-11-10 at 10 12 14 PM" src="https://user-images.githubusercontent.com/69999501/98771533-e2666400-23a1-11eb-97e2-95b3e85fa34a.png">

Model trained decently well and the results show

# Model Results
One example of the model running test data from the MNIST dataset
Array is set up in a way that the index that is returned is actually the number.
Down below you I have ploted the data at the random index that I have chosen from the dataset, and you can see that it is showing a 7.
In the next line, you can see that the output in is index 7, which is what we'd expect, and matches up with the plot.
<img width="649" alt="Screen Shot 2020-11-10 at 10 12 28 PM" src="https://user-images.githubusercontent.com/69999501/98771534-e2fefa80-23a1-11eb-8e93-a782eb6938de.png">
