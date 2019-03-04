%% Machine Learning: Programming Exercise 3
%% Multi-class Classification and Neural Networks
% In this exercise, you will implement one-vs-all logistic regression and neural 
% networks to recognize hand-written digits.
%% Files needed for this exercise
% * |ex3.mlx| - MATLAB Live Script that steps you through the exercise
% * |ex3data1.mat| - Training set of hand-written digits
% * |ex3weights.mat| - Initial weights for the neural network exercise
% * |submit.m |- Submission script that sends your solutions to our servers
% * |displayData.m| - Function to help visualize the dataset
% * |fmincg.m| - Function minimization routine (similar to |fminunc|)
% * |sigmoid.m| - Sigmoid function
% * |*lrCostFunction.m| - Logistic regression cost function
% * |*oneVsAll.m| - Train a one-vs-all multi-class classifier
% * |*predictOneVsAll.m| - Predict using a one-vs-all multi-class classifier
% * |*predict.m |- Neural network prediction function
% 
% _**indicates files you will need to complete*_
%% Confirm that your Current Folder is set correctly
% Click into this section, then click the 'Run section' button above. This will 
% execute the |dir| command below to list the files in your Current Folder. The 
% output should contain all of the files listed above and the 'lib' folder. If 
% it does not, right-click the 'ex3' folder and select 'Open' before proceding 
% or see the instructions in |README.mlx| for more details.

dir
%% Before you begin
% The workflow for completing and submitting the programming exercises in MATLAB 
% Online differs from the original course instructions. Before beginning this 
% exercise, make sure you have read through the instructions in |README.mlx| which 
% is included with the programming exercise files. |README| also contains solutions 
% to the many common issues you may encounter while completing and submitting 
% the exercises in MATLAB Online. Make sure you are following instructions in 
% |README| and have checked for an existing solution before seeking help on the 
% discussion forums.
%% 
%% 1. Multi-class Classification
% For this exercise, you will use logistic regression and neural networks to 
% recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition 
% is widely used today - from recognizing zip codes (postal codes) on mail envelopes 
% to recognizing amounts written on bank checks. This exercise will show you how 
% the methods you've learned can be used for this classication task. In the first 
% part of the exercise, you will extend your previous implemention of logistic 
% regression and apply it to one-vs-all classification.
%% 1.1 Dataset
% You are given a data set in |ex3data1.mat| that contains 5000 training examples 
% of handwritten digits*. The .mat format means that that the data has been saved 
% in a native MATLAB matrix format, instead of a text (ASCII) format like a csv-file. 
% These matrices can be read directly into your program by using the load command. 
% After loading, matrices of the correct dimensions and values will appear in 
% your program's memory. The matrix will already be named, so you do not need 
% to assign names to them. 
% 
% _*This is a subset of the MNIST <http://yann.lecun.com/exdb/mnist/) handwritten 
% digit dataset>_
% 
% Run the code below to load the data.

% Load saved matrices from file
load('ex3data1.mat');
% The matrices X and y will now be in your MATLAB environment
%% 
%     There are 5000 training examples in |ex3data1.mat|, where each training 
% example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is 
% represented by a floating point number indicating the grayscale intensity at 
% that location. The 20 by 20 grid of pixels is 'unrolled' into a 400-dimensional 
% vector. Each of these training examples becomes a single row in our data matrix 
% |X|. This gives us a 5000 by 400 matrix |X| where every row is a training example 
% for a handwritten digit image.
% 
% $$X=\left\lbrack \begin{array}{c}-{\left(x^{\left(1\right)} \right)}^T 
% -\\-{\left(x^{\left(2\right)} \right)}^T -\\\vdots \\-{\left(x^{\left(m\right)} 
% \right)}^T -\end{array}\right\rbrack$$
% 
%     The second part of the training set is a 5000-dimensional vector |y| 
% that contains labels for the training set. To make things more compatible with 
% MATLAB indexing, where there is no zero index, we have mapped the digit zero 
% to the value ten. Therefore, a '0' digit is labeled as '10', while the digits 
% '1' to '9' are labeled as '1' to '9' in their natural order.
%% 1.2 Visualizing the data
% You will begin by visualizing a subset of the training set. The code below 
% randomly selects selects 100 rows from |X| and passes those rows to the |displayData| 
% function. This function maps each row to a 20 pixel by 20 pixel grayscale image 
% and displays the images together. We have provided the |displayData| function, 
% and you are encouraged to examine the code to see how it works. After you run 
% this step, you should see an image like Figure 1.
% 
%                         

m = size(X, 1);
% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);
%% 1.3 Vectorizing logistic regression
% You will be using multiple one-vs-all logistic regression models to build 
% a multi-class classifier. Since there are 10 classes, you will need to train 
% 10 separate logistic regression classifiers. To make this training efficient, 
% it is important to ensure that your code is well vectorized. In this section, 
% you will implement a vectorized version of logistic regression that does not 
% employ any |for| loops. You can use your code in the last exercise as a starting 
% point for this exercise.
%% 1.3.1 Vectorizing the cost function
% We will begin by writing a vectorized version of the cost function. Recall 
% that in (unregularized) logistic regression, the cost function is
% 
% $$J(\theta) =\frac{1}{m}\sum_{i=1}^m{\left[-y^{(i)} \log(h_{\theta}(x^{(i)}))- 
% (1-y^{(i)}) \log(1- h_{\theta}(x^{(i)}))\right],$$
% 
% To compute each element in the summation, we have to compute $h_{\theta}(x^{(i)})$ 
% for every example $i$, where $h_{\theta}(x^{(i)}) = g(\theta^T x^{(i)})$ and 
% $g(z) =\frac{1}{1+e^{-z}}$ is the sigmoid function. It turns out that we can 
% compute this quickly for all our examples by using matrix multiplication. Let 
% us define $X$ and $\theta$ as
% 
% $$X\theta \text{ }=\left\lbrack \begin{array}{c}-{\left(x^{\left(1\right)} 
% \right)}^T \theta \text{ }-\\-{\left(x^{\left(2\right)} \right)}^T \theta \text{ 
% }-\\\vdots \\-{\left(x^{\left(m\right)} \right)}^T \theta \text{ }-\end{array}\right\rbrack 
% =\left\lbrack \begin{array}{c}-\text{ }\theta^T \left(x^{\left(1\right)} \right)-\\-{\text{ 
% }\theta }^T \left(x^{\left(2\right)} \right)\text{ }-\\\vdots \\-{\text{ }\theta 
% }^T \left(x^{\left(m\right)} \right)-\end{array}\right\rbrack \text{ }$$
% 
% In the last equality, we used the fact that $a^T b= b^T a$ if $a$ and $b$ 
% are vectors. This allows us to compute the products $\theta^T x^{(i)}$ for all 
% our examples $i$ in one line of code.
% 
%     Your job is to write the unregularized cost function in the file |lrCostFunction.m 
% |Your implementation should use the strategy we presented above to calculate 
% $\theta^T x^{(i)}$. You should also use a vectorized approach for the rest of 
% the cost function. A fully vectorized version of |lrCostFunction.m| should not 
% contain any loops. (Hint: You might want to use the element-wise multiplication 
% operation (|.*|) and the sum operation |sum| when writing this function)
%% 1.3.2 Vectorizing the gradient
% Recall that the gradient of the (unregularized) logistic regression cost is 
% a vector where the $j$th element is defined as 
% 
% $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m{\left( 
% h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}$$
% 
% To vectorize this operation over the dataset, we start by writing out all 
% the partial derivatives explicitly for all $\theta_j$,
% 
% $\left[ \begin{array}{c} \frac{\partial J(\theta)}{\partial \theta_0} \\\frac{\partial 
% J(\theta)}{\partial \theta_1} \\\frac{\partial J(\theta)}{\partial \theta_2} 
% \\\vdots \\\frac{\partial J(\theta)}{\partial \theta_n}    \end{array} \right] 
% = \frac{1}{m}\left[ \begin{array}{c} \sum_{i=1}^m{\left( h_\theta(x^{(i)})-y^{(i)}\right)x_0^{(i)} 
% \\\sum_{i=1}^m{\left( h_\theta(x^{(i)})-y^{(i)}\right)x_1^{(i)} \\\sum_{i=1}^m{\left( 
% h_\theta(x^{(i)})-y^{(i)}\right)x_2^{(i)} \\\vdots \\\sum_{i=1}^m{\left( h_\theta(x^{(i)})-y^{(i)}\right)x_n^{(i)}\end{array}\right] 
% =\frac{1}{m} \sum_{i=1}^m{\left( h_\theta(x^{(i)})-y^{(i)}\right)x^{(i)} =\frac{1}{m}X^T 
% {\left( h_\theta(x)-y\right)}$   (1)
% 
% where
% 
% $$h_{\theta \text{ }} \left(x\right)-y=\left\lbrack \begin{array}{c}h_{\theta 
% \text{ }} \left(x^{\left(1\right)} \right)-y^{\left(1\right)} \\h_{\theta \text{ 
% }} \left(x^{\left(2\right)} \right)-y^{\left(2\right)} \\\vdots \\h_{\theta 
% \text{ }} \left(x^{\left(m\right)} \right)-y^{\left(m\right)} \end{array}\right\rbrack$$
% 
%     Note that $x^{(i)}$ is a vector, while $(h_\theta(x^{(i)})-y^{(i)})$ 
% is a scalar (single number). To understand the last step of the derivation, 
% let $\beta_i = (h_\theta(x^{(i)})-y^{(i)})$ and observe that:
% 
% $$\sum_{i} \beta_i x^{\left(i\right)} =\left\lbrack \begin{array}{cccc}| 
% & | & \text{ } & |\\x^{\left(1\right)}  & x^{\left(2\right)}  & \dots  & x^{\left(m\right)} 
% \\| & | & \text{ } & |\end{array}\right\rbrack \left\lbrack \begin{array}{c}\beta_1 
% \\\beta_2 \\\vdots \\\beta_m \end{array}\right\rbrack =X^T \beta ,$$
% 
%     The expression above allows us to compute all the partial derivatives 
% without any loops. If you are comfortable with linear algebra, we encourage 
% you to work through the matrix multiplications above to convince yourself that 
% the vectorized version does the same computations. You should now implement 
% Equation (1) to compute the correct vectorized gradient. Once you are done, 
% complete the function |lrCostFunction.m| by implementing the gradient.
% 
% 
% 
% *Debugging Tip: *Vectorizing code can sometimes be tricky. One common strategy 
% for debugging is to print out the sizes of the matrices you are working with 
% using the size function. For example, given a data matrix $X$ of size 100 x 
% 20 (100 examples, 20 features) and $\theta$, a vector with dimensions 20 x 1, 
% you can observe that $X\theta$ is a valid multiplication operation, while $\theta 
% X$ is not. Furthermore, if you have a non-vectorized version of your code, you 
% can compare the output of your vectorized code and non-vectorized code to make 
% sure that they produce the same outputs.
%% 1.3.3 Vectorizing regularized logistic regression
% After you have implemented vectorization for logistic regression, you will 
% now add regularization to the cost function. Recall that for regularized logistic 
% regression, the cost function is defined as
% 
% $$J(\theta) =\frac{1}{m}\sum_{i=1}^m{\left[-y^{(i)} \log(h_{\theta}(x^{(i)}))- 
% (1 -y^{(i)}) \log(1-h_{\theta}(x^{(i)}))\right]+\frac{\lambda}{2m}\sum_{j=1}^n{\theta_j^2}$$
% 
% Note that you should not be regularizing $\theta_0$ which is used for the 
% bias term. Correspondingly, the partial derivative of regularized logistic regression 
% cost for $\theta_j$ is defined as
% 
% $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m{\left( 
% h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}\qquad \mathrm{for}\;j=0,$$
% 
% $$\frac{\partial J(\theta)}{\partial \theta_j} = \left( \frac{1}{m}\sum_{i=1}^m{\left( 
% h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}} \right)+\frac{\lambda}{m}\theta_j\qquad 
% \mathrm{for}\;j\geq1$$
% 
% Now modify your code in |lrCostFunction| to account for regularization. 
% Once again, you should not put any loops into your code. When you are finished, 
% run the code below to test your vectorized implementation and compare to expected 
% outputs:

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('Cost: %f | Expected cost: 2.534819\n',J);
fprintf('Gradients:\n'); fprintf('%f\n',grad);
fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003');
%% 
% *MATLAB Tip:* When implementing the vectorization for regularized logistic 
% regression, you might often want to only sum and update certain elements of 
% $\theta$. In MATLAB, you can index into the matrices to access and update only 
% certain elements. For example, |A(:,3:5)= B(:, 1:3)| will replace columns 3 
% to 5 of |A| with the columns 1 to 3 from |B|. One special keyword you can use 
% in indexing is the |end| keyword in indexing. This allows us to select columns 
% (or rows) until the end of the matrix. For example, |A(:, 2:end)| will only 
% return elements from the 2nd to last column of |A|. Thus, you could use this 
% together with the |sum| and |.^| operations to compute the sum of only the elements 
% you are interested in (e.g. |sum(z(2:end).^2))|. In the starter code,| lrCostFunction.m|, 
% we have also provided hints on yet another possible method computing the regularized 
% gradient.
% 
% _You should now submit your solutions.  Enter |_*submit|_* at the command 
% prompt, then enter or confirm your login and token when prompted._
%% 1.4 One-vs-all classication
% In this part of the exercise, you will implement one-vs-all classification 
% by training multiple regularized logistic regression classifiers, one for each 
% of the $K$ classes in our dataset (Figure 1). In the handwritten digits dataset, 
% $K = 10$, but your code should work for any value of $K$. 
% 
%     You should now complete the code in |oneVsAll.m| to train one classifier 
% for each class. In particular, your code should return all the classifier parameters 
% in a matrix $\Theta \in \mathbb{R}^{K\times(N+1)}$, where each row of $\Theta$ 
% corresponds to the learned logistic regression parameters for one class. You 
% can do this with a |for| loop from 1 to |K|, training each classifier independently.
% 
% Note that the |y| argument to this function is a vector of labels from 
% 1 to 10, where we have mapped the digit '0' to the label 10 (to avoid confusions 
% with indexing). When training the classifier for class $k \in \{1,\ldots,K\},$ 
% you will want a $m$-dimensional vector of labels $y,$ where $y_j\in \{0,\;1\}$ 
% indicates whether the $j$-th training instance belongs to class $k\; (y_j = 
% 1)$, or if it belongs to a different class $(y_j = 0).$ You may find logical 
% arrays helpful for this task.
% 
% *MATLAB Tip*: Logical arrays in MATLAB are arrays which contain binary 
% (0 or 1) elements. In MATLAB, evaluating the expression |a == b| for a vector 
% |a| (of size $m\times1$) and scalar |b| will return a vector of the same size 
% as |a| with ones at positions where the elements of |a| are equal to |b| and 
% zeroes where they are different. To see how this works for yourself, run the 
% following code:

a = 1:10; % Create a and b
b = 3;
disp(a == b) % You should try different values of b here
%% 
% Furthermore, you will be using |fmincg| for this exercise (instead of 
% |fminunc|). |fmincg| works similarly to |fminunc|, but is more more efficient 
% for dealing with a large number of parameters. After you have correctly completed 
% the code for |oneVsAll.m|, run the code below to use your |oneVsAll| function 
% to train a multi-class classifier.

num_labels = 10; % 10 labels, from 1 to 10 
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
%% 
% _You should now submit your solutions.  Enter |_*submit|_* at the command 
% prompt, then enter or confirm your login and token when prompted._
%% 1.4.1 One-vs-all prediction
% After training your one-vs-all classifier, you can now use it to predict the 
% digit contained in a given image. For each input, you should compute the 'probability' 
% that it belongs to each class using the trained logistic regression classifiers. 
% Your one-vs-all prediction function will pick the class for which the corresponding 
% logistic regression classifier outputs the highest probability and return the 
% class label $(1, 2,..., \text{ or } K)$ as the prediction for the input example.
% 
%     You should now complete the code in |predictOneVsAll.m| to use the 
% one-vs-all classifier to make predictions. Once you are done, run the code below 
% to call your |predictOneVsAll| function using the learned value of $\Theta$. 
% You should see that the training set  accuracy is about 94.9% (i.e., it classifies 
% 94.9% of the examples in the training set correctly).

pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
%% 
% _You should now submit your solutions.  Enter |_*submit|_* at the command 
% prompt, then enter or confirm your login and token when prompted._
%% 2. Neural Networks
% In the previous part of this exercise, you implemented multi-class logistic 
% regression to recognize handwritten digits. However, logistic regression cannot 
% form more complex hypotheses as it is only a linear classier. (You could add 
% more features such as polynomial features to logistic regression, but that can 
% be very expensive to train_.) _In this part of the exercise, you will implement 
% a neural network to recognize handwritten digits using the same training set 
% as before. The neural network will be able to represent complex models that 
% form non-linear hypotheses. 
% 
%     For this week, you will be using parameters from a neural network that 
% we have already trained. Your goal is to implement the feedforward propagation 
% algorithm to use our weights for prediction. In next week's exercise, you will 
% write the backpropagation algorithm for learning the neural network parameters.
%% 2.1 Model representation
% Our neural network is shown in Figure 2. It has 3 layers- an input layer, 
% a hidden layer and an output layer. Recall that our inputs are pixel values 
% of digit images. Since the images are of size 20 x 20, this gives us 400 input 
% layer units (excluding the extra bias unit which always outputs +1). As before, 
% the training data will be loaded into the variables |X| and |y|.
% 
% 
% 
%     You have been provided with a set of network parameters $(\Theta^{(1)},\Theta^{(2)})$ 
% already trained by us. These are stored in |ex3weights.mat| and are loaded into 
% |Theta1| and |Theta2 |by running the code below. The parameters have dimensions 
% that are sized for a neural network with 25 units in the second layer and 10 
% output units (corresponding to the 10 digit classes).

load('ex3data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

% Load saved matrices from file
load('ex3weights.mat'); 
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
%% 2.2 Feedforward propagation and prediction
% Now you will implement feedforward propagation for the neural network. You 
% will need to complete the code in |predict.m| to return the neural network's 
% prediction. You should implement the feedforward computation that computes $h_\theta(x^{(i)})$ 
% for every example $i$ and returns the associated predictions. Similar to the 
% one-vs-all classication strategy, the prediction from the neural network will 
% be the label that has the largest output $(h_\theta(x))_k$.
% 
% 
% 
% *Implementation Note: *The matrix |X| contains the examples in rows. When 
% you complete the code in |predict.m|, you will need to add the column of 1's 
% to the matrix. The matrices |Theta1| and |Theta2| contain the parameters for 
% each unit in rows. Specically, the first row of |Theta1| corresponds to the 
% first hidden unit in the second layer. In MATLAB, when you compute $z^{(2)} 
% = \Theta^{(1)}a^{(1)}$, be sure that you index (and if necessary, transpose) 
% |X| correctly so that you get $a^{(1)}$ as a column vector.
% 
% 
% 
% Once you are done, run the code below to call your predict function using 
% the loaded set of parameters for |Theta1| and |Theta2|. You should see that 
% the accuracy is about 97.5%.

pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
%% 
% The code below will displaying images from the training set one at a time, 
% while the console prints out the predicted label for the displayed image. Rerun 
% to repeat with another image.

%  Randomly permute examples
rp = randi(m);
% Predict
pred = predict(Theta1, Theta2, X(rp,:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
% Display 
displayData(X(rp, :));   
%% 
% _You should now submit your solutions.  Enter |_*submit|_* at the command 
% prompt, then enter or confirm your login and token when prompted._
%% Submission and Grading
% After completing this assignment, be sure to use the submit function to submit 
% your solutions to our servers. The following is a breakdown of how each part 
% of this exercise is scored.
% 
% 
% 
% You are allowed to submit your solutions multiple times, and we will take 
% only the highest score into consideration.