from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np

# break into tokens
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# process the tokens and produce embeddings
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # take mean to get a single embedding vector for the whole sentece and return
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding

# load in the sentiment analysis model, a pre-trained model and tokenizer
# pipeline handles processing and other details
sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)

def get_sentiment(text):
    # ulatimtely used for the pleasure (P) signal in PAD primarily
    # get a postive/negative label and confidence score
    result = sentiment_model(text)[0]
    label = result['label']
    score = result['score']

    # use the score and label to produce a single numeric sentiment value
    # positive will be a +score and negative will be -score
    sentiment_value = score if label == 'POSITIVE' else -score
    return sentiment_value # range [-1, 1]

# load in the emotion recognition model using transformers pipeline
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=-1)

def get_emotion(text):
    # get the list of emotions with their scores
    results = emotion_model(text)[0]

    # choose the top emotion and return (label, score)
    top = max(results, key=lambda r: r['score'])
    return top['label'], top['score']

# prepare and combine the features using the above definitions
def prep_features(text):
    embedding = get_embedding(text).flatten() # to make 1D for use in model
    sentiment = np.array([get_sentiment(text)]) # to make array for concatenation
    label, score = get_emotion(text)
    emotion = np.array([score])  # to make array for concatenation, only using the score but LUT could be used for more complex features

    features = np.concatenate([embedding, sentiment, emotion]) # size 770 vector, 768 + 1 + 1
    # features model making the text samples a 770 character numerical representation
    return features