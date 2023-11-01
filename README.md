# Med-ViLT

## Introduction
Visual Question Answering (VQA) has gained significant attention in recent years as it combines computer vision and natural language processing to enable machines to answer questions about images. In the medical field, VQA holds the potential to assist healthcare professionals in diagnosing and understanding radiology images.
This repo presents a detailed exploration of a Visual Question Answering model fine-tuned on the VQA-RAD dataset using the Vision-and-Language Transformer (ViLT). We explore this task's essential components and processes, including dataset preparation, model architecture, training procedures, and evaluation metrics. The proposed model demonstrates its potential to answer questions about medical radiology images with reasonable accuracy.

## Usage 
Install dependencies from requirements.txt and get dataset from [VQA-RAD](https://osf.io/89kps/). </br>
For inference use notebook in evals folder. </br>
For replication of model use the train-split with final_script.py to train the model.</br>

## Summary
Entire summary of the work done can be found [here](https://github.com/patrick-batman/Med-ViLT/blob/c99adea69f1d435d807289acf685a2a1ccc26112/summary.pdf).

## Data Augmentation
The VQA-RAD dataset, with 3515 data points, underwent crucial augmentation. Image techniques (flipping, blur, RAND-Augment) enhanced robustness. Question augmentation involved paraphrasing, transformer models, and innovative image-captioning methods for diverse question-answer pairs, fostering improved vision-language training through dataset enrichment.


## Model Architecture
### Soft-Encoding
We leveraged a pre-trained ViLT model from VQAv2 and sci-BERT embeddings, yielding better results with our limited dataset. While testing a transformer's sciBERT decoder, limited insights led us to abandon this approach. Addressing the prevalence of yes/no answers, we compiled a medical and radiology terms dictionary via web scraping, implementing a soft encoding method. Equal weight allocation to words in multi-word answers improved the model's response diversity. This tactic extended the model's capacity beyond multiclass classification, enabling it to handle open-set questions more effectively, ultimately enhancing its ability to generate varied answers in medical and radiology contexts.

### Contrastive Loss
Implementing a supervised contrastive loss function within a Visual Question Answering (VQA) system offers significant benefits. This method enhances the model's understanding of relationships between images and textual questions by creating more distinct embeddings through explicit definition of positive and negative pairs. It aids in capturing detailed associations between different modalities, addressing data imbalance issues and reducing sensitivity to answer distribution variations. The learned embeddings not only benefit various vision-language tasks but also act as regularization, curbing overfitting and improving adaptability to new data. This approach supports faster adaptation in fine-tuning, few-shot learning, and diverse scenarios requiring image-text comprehension.

## Results
### Closed Set Questions (From the test set)
<img width="795" alt="Screenshot 2023-11-02 at 12 21 57 AM" src="https://github.com/patrick-batman/Med-ViLT/assets/98745676/015944e6-7354-4193-aec1-3e656a8b3df3">

### Open Ended Questions
<img width="795" alt="Screenshot 2023-11-02 at 12 23 04 AM" src="https://github.com/patrick-batman/Med-ViLT/assets/98745676/4ac41386-9940-4910-bf8d-314f68f76ab5">

