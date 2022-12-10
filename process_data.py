import pandas as pd
# get review and rating from data tsv files
# save review and rating to data csv files
def preprocess_data(tsv_file):
    # read data tsv file, only keep review and rating columns
    df = pd.read_csv(tsv_file, sep='\t', usecols=['review', 'rating'])
    # convert rating to 0 or 1
    df['rating'] = df['rating'].apply(lambda x: 1 if x > 5 else 0)
    return df

# split train csv to train and validation csv
def split_train_val(train_csv):
    # read train csv file
    df = pd.read_csv(train_csv)
    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    # split train and validation
    train_df = df[:int(0.9*len(df))]
    # shape of train data
    print("Shape of train data: ", train_df.shape)
    val_df = df[int(0.9*len(df)):]
    # shape of validation data
    print("Shape of validation data: ", val_df.shape)
    # save to csv file
    train_df.to_csv('final_data/train.csv', index=False)
    val_df.to_csv('final_data/valid.csv', index=False)
if __name__ == '__main__':
    # preprocess data
    train_data = preprocess_data('drug_data/drugsComTrain_raw.tsv')
    test_data = preprocess_data('drug_data/drugsComTest_raw.tsv')
    # print first 5 rows of train data
    print(train_data.head())
    # print shape 
    print("Shape of train data: ", train_data.shape)
    print("Shape of test data: ", test_data.shape)
    # split train data to train and validation
    # save to csv file
    train_data.to_csv('drug_data/train.csv', index=False)
    split_train_val('drug_data/train.csv')
    test_data.to_csv('final_data/test.csv', index=False)
    