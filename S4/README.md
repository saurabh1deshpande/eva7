S4 Assignment QnA
-----------
PART 1
------
Screenshot of excel file
-----------------------

![image](https://user-images.githubusercontent.com/10797988/137641020-01784a4a-d46c-47c3-b7af-80ea9ef40951.png)

Steps
--------
Loop 1, 2, 3 for n times
1. Forward Pass : Understanding total loss given the current weights
		
- h1=w1 * i1 + w2 * i2			
- h2=w3 * i1 + w4 * i2			
- a_h1 = σ(h1) =1/(1+exp(-h1))			
- a_h2 = σ(h2) =1/(1+exp(-h2))			
- o1 = w5 * a_h1 + w6 * a_h2 			
- o2 = w7 * a_h1 + w8 * a_h2 			
- a_o1 = σ(o1) =1/(1+exp(-o1))			
- a_o2 = σ(o2) =1/(1+exp(-o2))			
- E1 = 0.5* (t1-a_o1)^2			
- E2 = 0.5* (t2-a_o2)^2			
- E_t = E1 + E2			

2. Finding partial derivative of the total loss with respect to weights
- 𝜕E_t/𝜕w1 = [(a_o1-t1) * a_o1*(1-a_o1)*w5+(a_o2-t2) * a_o2*(1-a_o2)*w7]*[a_h1*(1-a_h1)]*[i1]
- 𝜕E_t/𝜕w2 = [(a_o1-t1) * a_o1*(1-a_o1)*w5+(a_o2-t2) * a_o2*(1-a_o2)*w7]*[a_h1*(1-a_h1)]*[i2]
- 𝜕E_t/𝜕w3 = [(a_o1-t1) * a_o1*(1-a_o1)*w6+(a_o2-t2) * a_o2*(1-a_o2)*w8]*[a_h2*(1-a_h2)]*[i1]
- 𝜕E_t/𝜕w4 = [(a_o1-t1) * a_o1*(1-a_o1)*w6+(a_o2-t2) * a_o2*(1-a_o2)*w8]*[a_h2*(1-a_h2)]*[i2]
- 𝜕E_t/𝜕w5 =(a_o1-t1) *a_o1*(1-a_o1)*a_h1
- 𝜕E_t/𝜕w6 =(a_o1-t1) *a_o1*(1-a_o1)*a_h2
- 𝜕E_t/𝜕w7 =(a_o2-t2) *a_o2*(1-a_o2)*a_h1
- 𝜕E_t/𝜕w8 =(a_o2-t2) *a_o2*(1-a_o2)*a_h2

3. Updating the weights 
-       wi = wi - η * (𝜕E_t/𝜕wi) , where i = 1,2,3,4,5,6,7,8


Screenshots of Learning Rate changes
-----------------
Learning Rate = 0.1
![image](https://user-images.githubusercontent.com/10797988/137640733-ea9f3609-293a-4c25-b9a4-3bd20a885ab7.png)

Learning Rate = 0.2
![image](https://user-images.githubusercontent.com/10797988/137640811-0a82348d-3954-4dc7-b4a8-257ca92d654c.png)

Learning Rate = 0.5
![image](https://user-images.githubusercontent.com/10797988/137640859-60c70693-1d8c-4ad4-855e-9cd9bbfa146f.png)

Learning Rate = 0.8
![image](https://user-images.githubusercontent.com/10797988/137640883-8aec12b6-3613-4d8f-b38f-b26413c34c12.png)

Learning Rate = 1.0
![image](https://user-images.githubusercontent.com/10797988/137640914-21660ebb-109e-40ba-a631-e71ba1ac04cb.png)

Learning Rate = 2.0
![image](https://user-images.githubusercontent.com/10797988/137640953-90eec360-bbf8-4f79-bd65-92c012d3246b.png)



PART 2
------

The intention is that for MNIST achieve

1. 99.4% validation accuracy : achieved 99%
2. Less than 20k Parameters : 25.3K 
3. Less than 20 Epochs : OK

Network
-------

- Use of Conv layers (with and without padding), 
- use of BN (functional) : to make sure that the features available to the next layer is good, at - least 2 Conv layers away from output
- use of max pooling : used twice, at - least 2 Conv layers away from output
- use of GAP : instead of FC layers
- use of 1x1 : to reduce the number of channels

        self.conv1 = nn.Conv2d(1, 4, 3, padding=1) #input -1x28x28 Output - 4x28x28 
        # BN applied after this
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1) #input -4x28x28  Output - 8x28x28 
        # BN applied after this
        self.pool1 = nn.MaxPool2d(2, 2) #input -8x28x28 Output - 8x14x14 
        # drop out applied
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1) #input -8x14x14 Output - 16x14x14 
        # BN applied after this
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)#input -16x14x14  Output - 32x14x14  
        # BN applied after this
        self.pool2 = nn.MaxPool2d(2, 2) #input -32x14x14 Output - 32x7x7 
        # dropout applied after this
        self.conv5 = nn.Conv2d( 32, 64, 3) #input -32x7x7 Output - 64x5x5 
        # BN applied after this
        self.conv6 = nn.Conv2d(64, 10, 1) #input -64x5x5 Output - 10x5x5
        self.gap = nn.AvgPool2d(5) # input - 10x5x5 Output -10x1x1
        
Parameters
----------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #

            Conv2d-1      |      [-1, 4, 28, 28]        |      40
            Conv2d-2      |       [-1, 8, 28, 28]       |         296
         MaxPool2d-3      |       [-1, 8, 14, 14]       |         0
            Conv2d-4      |       [-1, 16, 14, 14]      |       1,168
            Conv2d-5      |      [-1, 32, 14, 14]       |       4,640
         MaxPool2d-6      |         [-1, 32, 7, 7]      |           0
            Conv2d-7      |         [-1, 1, 7, 7]      |         289
            Conv2d-8      |         [-1, 64, 7, 7]      |        640
            Conv2d-9      |         [-1, 64, 7, 7]      |        650
         AvgPool2d-10      |         [-1, 10, 1, 1]      |          0
----------------------------------------------------------------
- Total params: 7,723
- Trainable params: 7,723
- Non-trainable params: 0



Logs
--------epoch-1-------
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:52: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.5330947041511536 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.06it/s]

Test set: Average loss: 0.5182, Accuracy: 8684/10000 (87%)

--------epoch-2-------
loss=0.25590646266937256 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.33it/s]

Test set: Average loss: 0.2890, Accuracy: 9260/10000 (93%)

--------epoch-3-------
loss=0.15118540823459625 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.09it/s]

Test set: Average loss: 0.2004, Accuracy: 9499/10000 (95%)

--------epoch-4-------
loss=0.197140172123909 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.30it/s]

Test set: Average loss: 0.1822, Accuracy: 9540/10000 (95%)

--------epoch-5-------
loss=0.14860746264457703 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.12it/s]

Test set: Average loss: 0.1735, Accuracy: 9538/10000 (95%)

--------epoch-6-------
loss=0.3400249779224396 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.57it/s]

Test set: Average loss: 0.1494, Accuracy: 9584/10000 (96%)

--------epoch-7-------
loss=0.10430436581373215 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.32it/s]

Test set: Average loss: 0.1445, Accuracy: 9611/10000 (96%)

--------epoch-8-------
loss=0.174235999584198 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.31it/s]

Test set: Average loss: 0.1307, Accuracy: 9646/10000 (96%)

--------epoch-9-------
loss=0.09887590259313583 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.08it/s]

Test set: Average loss: 0.1260, Accuracy: 9652/10000 (97%)

--------epoch-10-------
loss=0.10377618670463562 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.09it/s]

Test set: Average loss: 0.1243, Accuracy: 9656/10000 (97%)

--------epoch-11-------
loss=0.1754930466413498 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.29it/s]

Test set: Average loss: 0.1202, Accuracy: 9662/10000 (97%)

--------epoch-12-------
loss=0.06780201196670532 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.24it/s]

Test set: Average loss: 0.1230, Accuracy: 9642/10000 (96%)

--------epoch-13-------
loss=0.10379300266504288 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.33it/s]

Test set: Average loss: 0.1141, Accuracy: 9681/10000 (97%)

--------epoch-14-------
loss=0.02957306243479252 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.35it/s]

Test set: Average loss: 0.1051, Accuracy: 9681/10000 (97%)

--------epoch-15-------
loss=0.07723627239465714 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.18it/s]

Test set: Average loss: 0.1043, Accuracy: 9707/10000 (97%)

--------epoch-16-------
loss=0.039169713854789734 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.91it/s]

Test set: Average loss: 0.0998, Accuracy: 9719/10000 (97%)

--------epoch-17-------
loss=0.06776997447013855 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.16it/s]

Test set: Average loss: 0.0927, Accuracy: 9735/10000 (97%)

--------epoch-18-------
loss=0.15072183310985565 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.07it/s]

Test set: Average loss: 0.0934, Accuracy: 9723/10000 (97%)

--------epoch-19-------
loss=0.06374228745698929 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.92it/s]

Test set: Average loss: 0.0933, Accuracy: 9727/10000 (97%)

--------epoch-20-------
loss=0.05094410851597786 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.18it/s]

Test set: Average loss: 0.0922, Accuracy: 9725/10000 (97%)
