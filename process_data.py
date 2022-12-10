import pandas as pd
# get review and rating from data tsv files
# save review and rating to data csv files
def preprocess_data(tsv_file):
    # read data tsv file, only keep review and rating columns
    df = pd.read_csv(tsv_file, sep='\t', usecols=['review', 'rating'])
    # convert rating to 0 or 1
    df['rating'] = df['rating'].apply(lambda x: 1 if x > 5 else 0)
    return df

if __name__ == '__main__':
    # preprocess data
    train_data = preprocess_data('data/drugsComTrain_raw.tsv')
    test_data = preprocess_data('data/drugsComTest_raw.tsv')
    # print first 5 rows of train data
    print(train_data.head())
    # print shape 
    print("Shape of train data: ", train_data.shape)
    print("Shape of test data: ", test_data.shape)
    # save to csv file
    train_data.to_csv('data/drugsComTrain.csv', index=False)
    test_data.to_csv('data/drugsComTest.csv', index=False)
    