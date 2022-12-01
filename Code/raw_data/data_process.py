
import pandas as pd
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.tokenize import word_tokenize

def get_stop_words():
    """
    get stopword table
    """
    with open("sentiment_analysis_eng/stopwords/english") as f:
        stopwords = f.readlines()
        stopwords = [word.strip() for word in stopwords]
    return stopwords


def normalise_text (text):
    """
    process each line data
    """
    stopwords = get_stop_words()
    text = text.lower() # 转换成小写
    text = text.replace(r"\#","") # replaces hashtags
    text = text.replace(r"http\S+","URL")  # remove URL addresses
    text = text.replace(r"@","")
    text = text.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.replace("\s{2,}", " ")    
    text = word_tokenize(text) # 分词
    text = [wnl.lemmatize(word) for word in text] # 词性还原
    text = [word for word in text if word not in stopwords] # 去停用词
    text = ' '.join(text) # 还原成文本
    return text

def trans_label(label):
    """
    label data transformation
    """
    label_dic={
        "__label__1":0,
        "__label__2":1
    }
    return label_dic[label]
if __name__ == "__main__":

    input_path = "sentiment_analysis_eng/raw_data/book_reviews.csv"
    output_path = "sentiment_analysis_eng/raw_data/book_reviews_res.csv"
    train = pd.read_csv(input_path)
    train["text"] = train["text"].apply(normalise_text)
    train["label"] = train["label"].apply(trans_label)

    train.to_csv(output_path,sep="\t",index=False, encoding="utf_8", mode="w", header=True)

