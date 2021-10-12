# PyTorch Assignment 2.5

This assignment is modified MNIST problem. In addition to the digit recognition, MNIST neural 
network must have a random integer (between 0 and 9) as additional input. Along with the digit 
prediction from image, neural network should also predict the sum of the digit from the image and
 the random number.
 
 Lets see design and implementation details.
 ##### Data Loader
 First task is to modify the data loader. Along with MNIST image, it must return a random 
 interger between 0 and 9 (and of course also true image label and true sum). So a custom data 
 loader is written extending MNIST class. We need to only override \_\_get_item\_\_(). It returns
  the data in following format,
  
  a tuple - ((image, random_integer),(true_image_label, true_sum))
  
   image and true_image_label are retrieved by calling \_\_get_item\_\_() of the parent class. 
   random_integer is generated using __randrange(10)__ function of PyTorch. Then tru_sum can be 
   calculated from true_image_label and the random_integer.
   
 ##### Neural Network
 The neural network has two parts. The convolutional (and couple of fully connected layers) for 
 image priocessing and few fully connected layers for sum processing. The network class (_forward_ 
 method in particular) has two inputs. The image (in Batch) and the random number (also in batch)
 
 
 The image and the random number is combined in summaation processing part of the netwrok as 
 follows,
 1. The convolutional layers will process the image.
 2. The output size of the CNN part is 10 corresponding to 10 classes (0 to 9)
 3. It will be in the form of likelihood because of the use of softmax
 4. Take the argmax of the output of the softmax layer of CNN
 5. Convert it to one hot encoding using _F.one_hot_ function of the PyTorch
 6. So the size will be 1000x10 (for the batch of 1000)
 7. Convert input random number also to one hot encoded value.  Again the size is 1000x10
 8. Concat both using PyTorch cat function with dim=1 (column wise concatenation)
 9. Put it through multiple fully connected layers
 10. Lat layer is again softmax and output size is 20 as maximum summation of two single digit 
 numbers can be 19 (0 to 20)
 
 The forward method returns CNN part and summation part as tuple.
 
 #####Training and loss calculation
 Training loop will pass image and random number batch received from custom dataloader through 
 the network. It will receive predictions for the image label and the summation. A custom loss 
 function is written to canculate combined loss. It will accept ground truth and predictions for 
 image and summation part. For each part a loss is calculated using _F.nll_loss_ function 
 (negative log likelihood). The log likelihood is a natural choice and goes well with the softmax
  output. The average of two losses is returned from the 
 custom loss function 
 method.
 
 #####Accuracy and evaluation
 Accuracy is calculated as % of correct predictions. Also loss is plot for the evaluation.
 
 #####Sample Evaluation
 A method is written to run few samples and print the predictions for the evaluation. Following 
 are some sample results,
 
Image Predicted:3, Image Actual:3,Random Input Number:8,Sum Precicted:11, Sum Actual:11

Image Predicted:1, Image Actual:1,Random Input Number:3,Sum Precicted:4, Sum Actual:4

Image Predicted:0, Image Actual:0,Random Input Number:4,Sum Precicted:4, Sum Actual:4

Image Predicted:1, Image Actual:1,Random Input Number:2,Sum Precicted:0, Sum Actual:3

Image Predicted:0, Image Actual:0,Random Input Number:7,Sum Precicted:7, Sum Actual:7

Image Predicted:7, Image Actual:7,Random Input Number:6,Sum Precicted:0, Sum Actual:13

Image Predicted:4, Image Actual:4,Random Input Number:7,Sum Precicted:11, Sum Actual:11

Image Predicted:1, Image Actual:1,Random Input Number:9,Sum Precicted:10, Sum Actual:10

Image Predicted:1, Image Actual:1,Random Input Number:5,Sum Precicted:6, Sum Actual:6

Image Predicted:2, Image Actual:2,Random Input Number:2,Sum Precicted:4, Sum Actual:4

#####Training Logs
Train Epoch: 1 [640/60000 (1%)]	Loss: 2.601291

Train Epoch: 1 [1280/60000 (2%)]	Loss: 2.579573

Train Epoch: 1 [1920/60000 (3%)]	Loss: 2.499271

Train Epoch: 1 [2560/60000 (4%)]	Loss: 2.434573

Train Epoch: 1 [3200/60000 (5%)]	Loss: 2.280845

Train Epoch: 1 [3840/60000 (6%)]	Loss: 2.241239

Train Epoch: 1 [4480/60000 (7%)]	Loss: 2.197711

Train Epoch: 1 [5120/60000 (9%)]	Loss: 2.196783

Train Epoch: 1 [5760/60000 (10%)]	Loss: 1.987944

Train Epoch: 1 [6400/60000 (11%)]	Loss: 2.103459

Train Epoch: 1 [7040/60000 (12%)]	Loss: 1.836078

Train Epoch: 1 [7680/60000 (13%)]	Loss: 1.922453

Train Epoch: 1 [8320/60000 (14%)]	Loss: 1.950262

Train Epoch: 1 [8960/60000 (15%)]	Loss: 1.847073

Train Epoch: 1 [9600/60000 (16%)]	Loss: 1.682628

Train Epoch: 1 [10240/60000 (17%)]	Loss: 1.534135

Train Epoch: 1 [10880/60000 (18%)]	Loss: 1.679816

Train Epoch: 1 [11520/60000 (19%)]	Loss: 1.616724

Train Epoch: 1 [12160/60000 (20%)]	Loss: 1.684713

Train Epoch: 1 [12800/60000 (21%)]	Loss: 1.659285

Train Epoch: 1 [13440/60000 (22%)]	Loss: 1.557932

Train Epoch: 1 [14080/60000 (23%)]	Loss: 1.706274

Train Epoch: 1 [14720/60000 (25%)]	Loss: 1.570186

Train Epoch: 1 [15360/60000 (26%)]	Loss: 1.585022

Train Epoch: 1 [16000/60000 (27%)]	Loss: 1.212152

Train Epoch: 1 [16640/60000 (28%)]	Loss: 1.367958

Train Epoch: 1 [17280/60000 (29%)]	Loss: 1.496133

Train Epoch: 1 [17920/60000 (30%)]	Loss: 1.524735

Train Epoch: 1 [18560/60000 (31%)]	Loss: 1.526654

Train Epoch: 1 [19200/60000 (32%)]	Loss: 1.477821

Train Epoch: 1 [19840/60000 (33%)]	Loss: 1.331638

Train Epoch: 1 [20480/60000 (34%)]	Loss: 1.466976

Train Epoch: 1 [21120/60000 (35%)]	Loss: 1.551928

Train Epoch: 1 [21760/60000 (36%)]	Loss: 1.370420

Train Epoch: 1 [22400/60000 (37%)]	Loss: 1.417648

Train Epoch: 1 [23040/60000 (38%)]	Loss: 1.391826

Train Epoch: 1 [23680/60000 (39%)]	Loss: 1.526440

Train Epoch: 1 [24320/60000 (41%)]	Loss: 1.497705

Train Epoch: 1 [24960/60000 (42%)]	Loss: 1.393142

Train Epoch: 1 [25600/60000 (43%)]	Loss: 1.575748

Train Epoch: 1 [26240/60000 (44%)]	Loss: 1.618791

Train Epoch: 1 [26880/60000 (45%)]	Loss: 1.638739

Train Epoch: 1 [27520/60000 (46%)]	Loss: 1.327729

Train Epoch: 1 [28160/60000 (47%)]	Loss: 1.332340

Train Epoch: 1 [28800/60000 (48%)]	Loss: 1.413171

Train Epoch: 1 [29440/60000 (49%)]	Loss: 1.415403

Train Epoch: 1 [30080/60000 (50%)]	Loss: 1.169901

Train Epoch: 1 [30720/60000 (51%)]	Loss: 1.459959

Train Epoch: 1 [31360/60000 (52%)]	Loss: 1.565820

Train Epoch: 1 [32000/60000 (53%)]	Loss: 1.453258

Train Epoch: 1 [32640/60000 (54%)]	Loss: 1.588836

Train Epoch: 1 [33280/60000 (55%)]	Loss: 1.523929

Train Epoch: 1 [33920/60000 (57%)]	Loss: 1.110984

Train Epoch: 1 [34560/60000 (58%)]	Loss: 1.263297

Train Epoch: 1 [35200/60000 (59%)]	Loss: 1.327841

Train Epoch: 1 [35840/60000 (60%)]	Loss: 1.211015

Train Epoch: 1 [36480/60000 (61%)]	Loss: 0.992418

Train Epoch: 1 [37120/60000 (62%)]	Loss: 1.544891

Train Epoch: 1 [37760/60000 (63%)]	Loss: 1.534909

Train Epoch: 1 [38400/60000 (64%)]	Loss: 1.131405

Train Epoch: 1 [39040/60000 (65%)]	Loss: 1.345360

Train Epoch: 1 [39680/60000 (66%)]	Loss: 1.036449

Train Epoch: 1 [40320/60000 (67%)]	Loss: 1.131922

Train Epoch: 1 [40960/60000 (68%)]	Loss: 1.165909

Train Epoch: 1 [41600/60000 (69%)]	Loss: 1.503922

Train Epoch: 1 [42240/60000 (70%)]	Loss: 1.601086

Train Epoch: 1 [42880/60000 (71%)]	Loss: 1.453146

Train Epoch: 1 [43520/60000 (72%)]	Loss: 1.224708

Train Epoch: 1 [44160/60000 (74%)]	Loss: 1.104309

Train Epoch: 1 [44800/60000 (75%)]	Loss: 1.219226

Train Epoch: 1 [45440/60000 (76%)]	Loss: 1.179424

Train Epoch: 1 [46080/60000 (77%)]	Loss: 1.054514

Train Epoch: 1 [46720/60000 (78%)]	Loss: 1.133546

Train Epoch: 1 [47360/60000 (79%)]	Loss: 1.279637

Train Epoch: 1 [48000/60000 (80%)]	Loss: 1.317693

Train Epoch: 1 [48640/60000 (81%)]	Loss: 1.076989

Train Epoch: 1 [49280/60000 (82%)]	Loss: 1.020066

Train Epoch: 1 [49920/60000 (83%)]	Loss: 1.412489

Train Epoch: 1 [50560/60000 (84%)]	Loss: 0.786996

Train Epoch: 1 [51200/60000 (85%)]	Loss: 1.055205

Train Epoch: 1 [51840/60000 (86%)]	Loss: 1.321700

Train Epoch: 1 [52480/60000 (87%)]	Loss: 1.237515

Train Epoch: 1 [53120/60000 (88%)]	Loss: 1.261555

Train Epoch: 1 [53760/60000 (90%)]	Loss: 1.121562

Train Epoch: 1 [54400/60000 (91%)]	Loss: 1.380413

Train Epoch: 1 [55040/60000 (92%)]	Loss: 1.281522

Train Epoch: 1 [55680/60000 (93%)]	Loss: 1.160847

Train Epoch: 1 [56320/60000 (94%)]	Loss: 1.385437

Train Epoch: 1 [56960/60000 (95%)]	Loss: 1.120340

Train Epoch: 1 [57600/60000 (96%)]	Loss: 1.371146

Train Epoch: 1 [58240/60000 (97%)]	Loss: 1.544102

Train Epoch: 1 [58880/60000 (98%)]	Loss: 1.202944

Train Epoch: 1 [59520/60000 (99%)]	Loss: 1.264256

Test set: Avg. loss: 0.0008, Accuracy Image: 9554/10000 (96%) , Accuracy Sum: 5301/10000 (53%)   