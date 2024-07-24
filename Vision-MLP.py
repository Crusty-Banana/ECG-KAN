import timm
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from engine import train_one_epoch, evaluate

# Create a Vision Transformer (ViT) model
vit_model = timm.create_model(
    'vit_tiny_patch16_224',  # Use a ViT model from timm library
    pretrained=False,
    num_classes=6,
    drop_rate=0.0,
    drop_path_rate=0.05,
    img_size=224,
)

# Dataset transformations
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)

train_dir = "/home/LENOVO/code/ECG_Image_data/train"
test_dir = "/home/LENOVO/code/ECG_Image_data/test"

trainset = datasets.ImageFolder(root=train_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=144,
                                          shuffle=True, num_workers=2)

testset = datasets.ImageFolder(root=test_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=144, shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5')

# Optimizer and loss function
optimizer = optim.SGD(vit_model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
vit_model.to(device)

# Train using engine.py

for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        print(i)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = vit_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        
# Evaluate
test_stats = evaluate(testloader, vit_model, device=device)
print(f"Accuracy of the network on the {len(testset)} test images: {test_stats['acc1']:.1f}%")

print('Finished Training')
