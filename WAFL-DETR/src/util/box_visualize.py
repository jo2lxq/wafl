import torch, torchvision
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import json
import random


oid_labels = [
  'Person',
  'Bicycle',
  'Car',
  'Table',
  'Door',
  'Fire hydrant',
  'Waste container',
  'Ball',
  'Cat',
  'Drink'
]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    # (center_x, center_y, width, height)*N -> (center_x*N, center_y*N, width*N, height*N)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    # (center_x, center_y, width, height)*N の形に戻す
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_bboxes_from_outputs(outputs, im, threshold=0.75):
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  probas_to_keep = probas[keep]
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  return probas_to_keep, bboxes_scaled

def plot_finetuned_results(pil_img, image_name, threshold, prob=None, boxes=None, labels=None):
  plt.figure(figsize=(16, 10))
  plt.imshow(pil_img)
  ax = plt.gca()
  colors = COLORS * 100
  if prob is not None and boxes is not None:
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
      ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                 fill=False, color=c, linewidth=4))
      cl = p.argmax()
      print(labels, p)
      text = f'{labels[cl]}: {p[cl]:0.2f}'
      ax.text(xmin, ymin, text, fontsize=15,
              bbox=dict(facecolor='yellow', alpha=0.5))
  plt.axis('off')
  plt.savefig(f'./outputs/{image_name}_{device}_{epoch}_{threshold}.png')

# object detection
def run_worflow(my_image, image_name, my_model, labels, device, epoch, topology, threshold=0.75):
  # normalization of mean-std input images (batch size : 1)
  img = transform(my_image).unsqueeze(0)
  outputs = my_model(img)

  probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, my_image, threshold=threshold)
  plot_finetuned_results(my_image, image_name, threshold, probas_to_keep, bboxes_scaled, labels)

if __name__ == "__main__":
  model = torch.hub.load('facebookresearch/detr', 'detr_resnet50_dc5', pretrained=False)
  in_features = model.class_embed.in_features
  model.class_embed = nn.Linear(in_features=in_features, out_features=11)

  # please edit these configurations
  device = 5
  epoch = '0049'
  threshold = 0.75
  image_name = 'ade99cb3add1eb35.jpg'

  checkpoint = torch.load(f'./outputs/node{device}/checkpoint{epoch}.pth', map_location='cpu')
  model.load_state_dict(checkpoint['model'], strict=False)

  im = Image.open(f'../data/custom/val2017/{image_name}')
  run_worflow(im, image_name, model, oid_labels, device, epoch, threshold)
