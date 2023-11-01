import pandas as pd
from transformers import ViltConfig, ViltProcessor
import torch
from PIL import Image
import os
from transformers import ViltForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader
import json


final_data=pd.read_csv('/Users/raunakpandey/Documents/programming/projects/med-flamingo/ViLT/final_data.csv')
epochs = 50
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# read the encodings.json file
with open(os.path.join(os.getcwd(),'tokens.json'), 'r') as f:
    data = json.load(f)

# extract the unique_answers, answer_to_index, and predicted_answers lists
unique_answers = data['unique_answers']
answer_to_index = data['answer_to_index']
predicted_answers = data['predicted_answers']


print("recieved labels")


import torch
from PIL import Image
class VQADataset(torch.utils.data.Dataset):
  def __init__(self, processor):
    final_data = pd.read_csv('final_data.csv')
    self.questions = final_data['question']
    self.image_id = final_data['image']
    self.processor = processor
    self.answers = final_data['ans-hyp']

  def __len__(self):
    return len(self.image_id)

  def __getitem__(self,idx):
    # get image + text
    questions = self.questions[idx]
    img_path = os.path.join(os.getcwd(),'osfstorage-archive','VQA_RADImage',self.image_id[idx])
    image = Image.open(img_path)
    answer = self.answers[idx]
    
    encoding = self.processor(image, questions, padding="max_length", truncation=True, return_tensors="pt")

    # remove batch dimension
    for k,v in encoding.items():
      encoding[k] = v.squeeze()

    answer_words = str(answer).split('-')
    scores = list()
    if len(answer_words) > 1:
      scores = [(1/len(answer_words)) for _ in range(len(answer_words))]
    else: 
      scores = [1.0]
    # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
    targets = torch.zeros(len(answer_to_index))
    scores_final = torch.tensor(scores)
    for ans, score in zip(answer_words,scores_final):
      targets[answer_to_index[ans]] = score
    encoding["labels"] = targets
    return encoding

dataset = VQADataset(processor=processor)
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm")
model.classifier[3] = torch.nn.Linear(in_features=1536 , out_features=1421, bias=True)
model.to(device)

from torch.utils.data import DataLoader

def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  labels = [item['labels'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = torch.stack(labels)

  return batch

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)




optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    running_loss = 0.0
    for batch in tqdm(train_dataloader):
        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    if epoch%5 == 0:
            EPOCH = epoch
            PATH = os.path.join(os.getcwd(),f"model{epoch}.pt")
            LOSS = running_loss

            torch.save({
                        'epoch': EPOCH,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': LOSS,
                        }, PATH)  