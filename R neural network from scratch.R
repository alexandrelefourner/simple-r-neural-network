#Code example in R for a very basic neural network.

hidden_layer_node_size <- 64 #Size of the hidden layer
iterations <- 200000
learning_rate <- 0.01

X <- matrix( #Example matrix
  c(1,1,1,
    1,0,1,
    0,1,1,
    0,0,1),
  nrow= 4,
  ncol=3,
  byrow=TRUE)

Y <- matrix( #Label Matrix
  c(0,
    1,
    1,
    0),
  nrow= 4,
  ncol=1,
  byrow=TRUE)

function.sigmoid <- function(x) #Sigmoid function used as activation function
{
  return (1/(1+exp(-x)))
}
function.sigmoid_logistic_reverse_backprop <- function(x) #Sigmoid function used as activation function
{
  return ((1-x)*x)
}

set.seed(12345)

#hidden layer weights
hl <- matrix(
  c(runif(hidden_layer_node_size)*2-1,
    runif(hidden_layer_node_size)*2-1,
    runif(hidden_layer_node_size)*2-1),
  nrow = 3,
  ncol = hidden_layer_node_size)


#Output layer wrights
wo <- matrix(
  c(runif(hidden_layer_node_size)*2-1),
  ncol = 1)


d <- data.frame("Iteration" = 0, "Error" = 2)


for(i in seq(1,iterations)){
  #We forward propagate
  hidden_layer <- function.sigmoid(X %*% hl)
  output_layer <- function.sigmoid(hidden_layer %*% wo)
  
  SE_o <- (output_layer - Y) #We calculate the loss function
  
  
  #We calculate gradients (back propagation)
  output_layer_error_gradient = SE_o*function.sigmoid_logistic_reverse_backprop(output_layer)
  error_h = output_layer_error_gradient %*% t(wo)
  hidden_layer_error_gradient = error_h*function.sigmoid_logistic_reverse_backprop(hidden_layer)
  
  #We apply the change to our model
  wo <- wo - learning_rate * (t(hidden_layer) %*% output_layer_error_gradient)
  hl <- hl - learning_rate * (t(X) %*% hidden_layer_error_gradient)
  if(i %% 100 == 0){
    d <- rbind(d, list(i,sum(abs(SE_o)))) #Used for ploting purpose
  }
  
}

print(output_layer)
plot(d)
