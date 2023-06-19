import torch
import torch.nn as nn
import boto3
import json
from smart_open import open as smart_open
import io


class MLPRegressionBN(nn.Module):
    def __init__(self):
        super(MLPRegressionBN, self).__init__()

        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 10)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)

        return x


def lambda_handler(event, context):
    # get inputs
    for key, value in event.items():
        event[key] = float(value)

    input_vec = torch.FloatTensor([
        event['windSpeed'],
        event['sunshine'],
        event['humidity'],
        event['cloudLevel'],
        event['hour'],
        event['season'],
        event['weekday'],
        event['isHoliday'],
        event['feelingTemp'],
        event['discomfortIndex'],
        event['dustLevel'],
        event['isRain']
    ])

    # initiate model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPRegressionBN()

    # 분식, 족/보, 찜/탕, 치킨, 카페/디저트, 피자, 한식, 일식, 패스트푸드, 아시안
    category_name = ['snack', 'jokbo', 'jjim', 'chicken',
                     'cafe', 'pizza', 'korean', 'japanese', 'fastfood', 'asian']

    # get object from s3

    load_path = "s3://deliverus.online-regression-model/regressionMLP.pt"
    with smart_open(load_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        model.load_state_dict(torch.load(buffer, map_location=device))

    with torch.no_grad():
        model.eval()
        inputs = input_vec.unsqueeze(0)
        outputs = model(inputs).tolist()
        result = {}
        for i in range(len(category_name)):
            result[category_name[i]] = outputs[0][i]
        top_five = sorted(list(result.items()), key=lambda x: -x[1])[:5]
        print(top_five)

    res = json.dumps(top_five)

    return res
