# Experiments

## 1. Custom Network

### 1.1. Architecture

1
```python
#Example network
class CustomNet(nn.Module):
    def __init__(self):
        # In the constructor we will specify those processing blocks which require learnable parameters
        # We just define them to ensure persistence during training, although they will remain disconnected
        # until we define a processing pipeline in forward method.
        super(CustomNet, self).__init__()

        #2D convolution to operate over RGB images
        self.conv1 = nn.Conv2d(3, 16, 3)
        # Maxpooling with kernel 2x2 and stride=2
        self.pool = nn.MaxPool2d(3, 2)
        # 2D convolution
        self.conv2 = nn.Conv2d(16, 32, 3)
        # 2D convolution
        self.conv3 = nn.Conv2d(32, 64, 3)
        # 2D convolution
        self.conv4 = nn.Conv2d(64, 128, 3)
        # Fully-connected layer which expects a linearized vector input (after flattening)
        # and produces 120 channels at the output
        self.fc1 = nn.Linear(40000, 1000)
        # Fully-connected layer 
        self.fc2 = nn.Linear(1000, 128)
        # Fully-connected layer 
        self.fc3 = nn.Linear(128, 32)
        # Fully-connected layer 
        self.fc4 = nn.Linear(32, 2)

    #In forward method we connect layers and define the processing pipeline of the network
    def forward(self, x):
        #Convolutional Blocks => Conv -> Relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        #Flatten method vectorizes the spatial output of size hxwxchannels into a vector of length h*w*channels
        #by setting the parameter to 1, we start to flatten in dim=1 and do not vectorize the dimension representing
        #the images in the batch
        x = x.flatten(1)
        
        #Fully connected blocks => Linear -> Relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    

customNet = CustomNet()
customNet.to(device) #copy the network to the device (gpu)
#Loss function
criterion = nn.CrossEntropyLoss()
# SGD with momentum 
optimizer_ft = optim.SGD(customNet.parameters(), lr=1e-2, momentum=0.9)
# An lr strategy which decreases lr by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
```


Very poor performing, the model is not able to overfit, so we must increase the size and also check the learning rate

$AUC = 0.5237$, $Loss = 0.5657$

![img_1.png](media/img_1.png)

---
2 

```python
#Example network
class CustomNet(nn.Module):
    def __init__(self):
        # In the constructor we will specify those processing blocks which require learnable parameters
        # We just define them to ensure persistence during training, although they will remain disconnected
        # until we define a processing pipeline in forward method.
        super(CustomNet, self).__init__()

        #2D convolution to operate over RGB images
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.conv5 = nn.Conv2d(32, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.conv7 = nn.Conv2d(64, 128, 3)
        self.conv8 = nn.Conv2d(128, 128, 3)
        # Maxpooling with kernel 2x2 and stride=2
        self.pool = nn.MaxPool2d(2)
        # Fully-connected layer which expects a linearized vector input (after flattening)
        # and produces 120 channels at the output
        self.fc1 = nn.Linear(12800, 4096)
        # Fully-connected layer 
        self.fc2 = nn.Linear(4096, 1024)
        # Fully-connected layer 
        self.fc3 = nn.Linear(1024, 2)

    #In forward method we connect layers and define the processing pipeline of the network
    def forward(self, x):
        #Convolutional Blocks => Conv -> Relu -> pool
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv7(x)))))
        #Flatten method vectorizes the spatial output of size hxwxchannels into a vector of length h*w*channels
        #by setting the parameter to 1, we start to flatten in dim=1 and do not vectorize the dimension representing
        #the images in the batch
        x = x.flatten(1)
        
        #Fully connected blocks => Linear -> Relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
```

Now using adam optimizer with $lr=1e-4$, still have a plateu, will try with another scheduler

---
3
Now using the plateu we improve the results a little bit
$AUC = 0.5637$

4
Use num workers and pin_memory to speed up training

5
__


## 2. Fine-tuning

### 2.1. Architecture

1. AlexNet

A bit of overfitting, need to try regularization techniques and trian for more epchos
$$AUC = 0.6731$$, $$Loss = 0.5317$$

