import torch

def evaluate(net, valloader, device = torch.device("cpu"), step = 50):
	net.eval()	# 評価モードに設定
	correct = 0
	total = 0

	with torch.no_grad():  # 勾配計算を停止
		for batch_index, (inputs, labels) in enumerate(valloader):
			if batch_index == step:
				break
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = net(inputs)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		return correct / total * 100

