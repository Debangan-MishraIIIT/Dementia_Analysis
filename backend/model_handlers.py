import torch
from torchvision import transforms
import torch.nn.functional as F
from models import Network

def get_predictions(img, model):
    transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
    img= transform(img).unsqueeze(dim=0)
    logits_sf= F.softmax(model(img), dim=1)
    _, predicted = torch.max(logits_sf, 1)
    return predicted.item()

def get_model():
    model = Network(1, 2)
    model.load_state_dict(torch.load("./best_model.pth"))
    return model