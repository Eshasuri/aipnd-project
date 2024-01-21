import argparse
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim



def build_model(arch, hidden_units, learning_rate):
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError("Please select: vgg19/vgg16/alexnet")

    num_classes = len(image_datasets['train'].classes)
    
    if arch == 'vgg19' or arch == 'vgg16':
        in_features = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units, num_classes)
        )
        model.classifier = classifier

    elif arch == 'alexnet':
        in_features = model.classifier[1].in_features
        classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units, num_classes)
        )
        model.classifier = classifier
    return model, classifier


def save_checkpoint(model, hidden_units, num_epochs, optimizer, path):
    checkpoint = {
    'model_state_dict': model.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx,
    'optimizer_state_dict': optimizer.state_dict(),
    'hidden_units': hidden_units,
    'num_epochs': num_epochs
    }

    torch.save(checkpoint, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training part of COMMAND LINE APPLICATION")
    parser.add_argument("data_dir", type=str)
    parser.add_argument('--save_dir', type=str, action='store', default='checkpoint.pth')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg19', 'vgg16', 'alexnet'])
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    data_dir = args.data_dir
    path = 'checkpoint.pth'

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456,0.406], [0.229, 0.224, 0.225])
    ])
    }

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    image_datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32),
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    model, classifier = build_model(args.arch, args.hidden_units, args.learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(image_datasets['train'].classes)
    in_features = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, num_classes)
    )
    model.classifier = classifier

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate) 
    model.to(device)
    for epoch in range(args.num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            model.eval()
            validation_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in dataloaders['test']:
                    images, labels = images.to("cuda"), labels.to("cuda")
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            validation_accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {validation_loss/len(dataloaders["test"]):.4f}, Validation Accuracy: {validation_accuracy:.2f}%')
        save_checkpoint(model, args.hidden_units, args.num_epochs, optimizer, path)