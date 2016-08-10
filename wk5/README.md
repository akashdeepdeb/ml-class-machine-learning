
##Neural networks: learning
In this week you'll learn a really cool concept called backpropogation. Backpropogation is initially a little hard to understand, and admittedly, I don't understand it well enough myself, but it gets easier to get a grasp of after you've done the assignments and while you're doing it. In the earlier week, you directly use the weights of the neural network given to you in the assignment; here, you have to train your datasets to learn these weights and regularize it with a good choice of lambda to get an accuracy >95%

##Notes
- I have added 5 .png images for the visual of the weights of the hidden layer (25 x 400) excluding the weight of the bias unit. 
- Each has a jump in lambda by one order of magnitude to show how increasing or decreasing lambda beyond a certain value (in our case lambda = 3) can result in inaccuracy in the neural network. 
- I haven't changed MaxIter (the maximum number of iterations) since my costFunction takes too much time to compute. But by increasing the number of iterations you'd almost assuredly get a lower cost function and greater accuracy since now you can sample the cost function multiple times which makes your neural network more accurate. 


##What you'll learn
- Modifying the cost function for backpropogation
- Unrolling parameters and reshaping matrices
- Checking gradient using the derivatives
- Initializing weights randomly to break symmetry so that the NN can be trained
- How backpropogation is used in autonomous driving
  - first, when the vehicle moves, the input (X in our case) is in terms of what the camera captures ahead. ie. the road.
  - second, the output is how to driver moves the vehicle according to this input. 
  - after driving for a while, the neural network learns when to move left or right and applies it's trained weights to new data (or newer paths [how cool!])
