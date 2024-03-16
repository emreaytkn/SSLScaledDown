import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torchinfo import summary
from torchmetrics import Accuracy

import mlflow

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("DL Debugging")

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images.
"""

batch_size = 4
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(
    "../datasets/cifar10", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Get cpu or gpu for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


class SimpleCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)  # Input channel, output channel, kernel size
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes in CIFAR-10

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch dimension
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # if i >= 1:
        #    break

        X, y = data

        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model(X)
        loss = loss_fn(outputs, y)
        accuracy = metrics_fn(outputs, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 125 == 124:  # print every 100 mini-batches
            loss, current = loss.item(), i
            print(
                f"loss: {loss:3f} accuracy: {accuracy:3f} [{current} / {len(dataloader)}]"
            )
            mlflow.log_metric("loss", f"{running_loss / 2000:.3f}", step=(i // 2000))
            mlflow.log_metric("accuracy", f"{accuracy:3f}", step=(i // 2000))

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


net = SimpleCNN()

epochs = 2
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

with mlflow.start_run():
    params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "loss_function": criterion.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "SGD",
    }
    # Log training parameters.
    mlflow.log_params(params)

    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(net)))
    mlflow.log_artifact("model_summary.txt")

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, net, criterion, metric_fn, optimizer, t)

    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(net, "model")

print("Finished training.")
