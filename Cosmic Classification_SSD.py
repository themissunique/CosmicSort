import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
num_epochs = 10
learning_rate = 0.001
train_data, train_labels = ...
val_data, val_labels = ...
transform = transforms.Compose([
    transforms.ToTensor(),])
model = torchvision.models.detection.ssd513_v2(pretrained=True)
num_classes = len(set(train_labels))
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_data:
        images = transform(images)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
torch.save(model.state_dict(), 'cosmic_ray_classifier.pt')
test_image = Image.open('test_image.jpg')
test_image_tensor = transform(test_image)
with torch.no_grad():
    predictions = model([test_image_tensor])


