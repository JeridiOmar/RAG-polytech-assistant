import pandas as pd


def clean_text(text):
    return ' '.join(text.lower().split())


def clean_data(path):
    documents = pd.read_csv(path, header=0)
    for index, row in documents.iterrows():
        documents['content'][index] = clean_text(documents['content'][index])
    return documents


if __name__ == '__main__':
    data_path = input('Enter csv to clean:')
    result_path = input('Enter where to save:')
    cleaned = clean_data(data_path)
    cleaned.to_csv(result_path, encoding="utf-8", index=False)
